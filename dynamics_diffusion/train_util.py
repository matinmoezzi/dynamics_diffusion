import copy
import functools
import os
from pathlib import Path
import hydra

import blobfile as bf
import torch as th
import numpy as np
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
from tqdm import trange
from dynamics_diffusion.script_util import ConfigStore, create_ema_and_scales_fn
from dynamics_diffusion.sde_models.ema import ExponentialMovingAverage
from .mlp import MLP

from dynamics_diffusion.sde import SDE, VESDE, VPSDE, subVPSDE

from . import dist_util, logger
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler, create_named_schedule_sampler


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def _expand_tensor_shape(x, shape):
    while len(x.shape) < len(shape):
        x = x[..., None]
    return x


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        dataset,
        total_training_steps,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_amp,
        opt_name: str,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.dataset = dataset
        self.diffusion = diffusion
        if isinstance(model, th.nn.Module):
            self.model = model
        elif model.name == "MLP":
            tmp_data = next(self.dataset)
            assert (
                "state" in tmp_data[1] and "action" in tmp_data[1]
            ), "Dataset must have state and action"
            self.x_dim = tmp_data[1]["state"].shape[1]
            self.cond_dim = self.x_dim + tmp_data[1]["action"].shape[1]
            self.model = MLP(self.x_dim, self.cond_dim, learn_sigma=model.learn_sigma)

        self.model = self.model.to(self.local_rank)
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        if not isinstance(self.diffusion, SDE):
            schedule_sampler = schedule_sampler or UniformSampler(diffusion)
            self.schedule_sampler = create_named_schedule_sampler(
                schedule_sampler, self.diffusion
            )
        else:
            self.schedule_sampler = schedule_sampler
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = int(total_training_steps)

        self.use_amp = use_amp
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.opt_name = opt_name.lower()

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.opt = self._get_optimizer()

        if self.use_amp:
            self.scaler = th.cuda.amp.GradScaler()

        self.ema = [
            ExponentialMovingAverage(self.model.parameters(), decay=rate)
            for rate in range(len(self.ema_rate))
        ]

        self._load_checkpoint()

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(self.model, device_ids=[self.local_rank])
            self.model = self.ddp_model
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_checkpoint(self):
        self.resume_checkpoint = (
            self._find_resume_checkpoint() or self.resume_checkpoint
        )
        if self.resume_checkpoint:
            logger.log(
                f"loading model, ema, optimizer from checkpoint: {self.resume_checkpoint}..."
            )
            self.snapshot = th.load(
                self.resume_checkpoint, map_location=f"cuda:{self.local_rank}"
            )
            self.step = self.resume_step = self.snapshot["step"]
            self.model.load_state_dict(self.snapshot["model_state"])
            self.opt.load_state_dict(self.snapshot["opt_state"])
            for rate, ema in zip(self.ema_rate, self.ema):
                ema.load_state_dict(self.snapshot[f"ema_{rate}"])

    def _find_resume_checkpoint(self):
        # On your infrastructure, you may want to override this to automatically
        # discover the latest checkpoint on your blob storage, etc.
        train_path = Path(self.resume_checkpoint, "train")
        if train_path.is_dir():
            list_models = list(train_path.glob("checkpoint_*.pt"))
            return str(max(list_models, key=os.path.getctime))
        return None

    def run_loop(self):
        assert (
            self.total_training_steps > self.step
        ), "total_training_steps must be greater than step"
        for _ in trange(
            self.step,
            self.total_training_steps,
            initial=self.step,
            total=self.total_training_steps,
        ):
            if not (not self.lr_anneal_steps or self.step < self.lr_anneal_steps):
                break
            batch, cond = next(self.dataset)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.local_rank == 0 and self.step % self.save_interval == 0:
                self.save()
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if self.local_rank == 0 and (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_amp:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        for ema in self.ema:
            ema.update(self.model.parameters())
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(self.local_rank)
            micro_cond = {
                k: v[i : i + self.microbatch].to(self.local_rank)
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.local_rank)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            with th.set_grad_enabled(True), th.amp.autocast(
                device_type="cuda", dtype=th.float16, enabled=self.use_amp
            ):
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self):
        snapshot = {}
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot["model_state"] = raw_model.state_dict()
        snapshot["opt_state"] = self.opt.state_dict()
        snapshot["step"] = self.step
        for rate, ema in zip(self.ema_rate, self.ema):
            snapshot[f"ema_{rate}"] = ema.state_dict()

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"checkpoint_{self.step:06d}.pt"),
            "wb",
        ) as f:
            th.save(snapshot, f)

    def _get_optimizer(self):
        if self.opt_name == "adam":
            return th.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        elif self.opt_name == "adamax":
            return th.optim.Adamax(
                self.model.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        elif self.opt_name == "adamw":
            return th.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        elif self.opt_name == "rmsprop":
            return th.optim.RMSprop(
                self.model.parameters(),
                lr=self.lr,
                alpha=self.alpha,
                eps=self.eps,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                centered=self.centered,
            )
        elif self.opt_name == "sgd":
            return th.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                dampening=self.dampening,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.opt_name}")


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


class CMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        target_ema_mode,
        start_ema,
        scale_mode,
        start_scales,
        end_scales,
        distill_steps_per_iter,
        teacher_model_path,
        teacher_dropout,
        training_mode,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_mode = training_mode

        self.ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=target_ema_mode,
            start_ema=start_ema,
            scale_mode=scale_mode,
            start_scales=start_scales,
            end_scales=end_scales,
            total_steps=self.total_training_steps,
            distill_steps_per_iter=distill_steps_per_iter,
        )

        cfg = ConfigStore.get_config()
        if len(teacher_model_path) > 0:  # path to the teacher score model.
            logger.log(f"loading the teacher model from {teacher_model_path}")
            self.teacher_model = hydra.utils.instantiate(
                cfg.trainer.model, dropout=teacher_dropout, distillation=False
            )
            self.teacher_diffusion = hydra.utils.instantiate(cfg.trainer.diffusion)

            self.teacher_model.load_state_dict(
                dist_util.load_state_dict(teacher_model_path, map_location="cpu"),
            )

            self.teacher_model.to(self.local_rank)
            self.teacher_model.eval()

            for dst, src in zip(
                self.model.parameters(), self.teacher_model.parameters()
            ):
                dst.data.copy_(src.data)

        else:
            self.teacher_model = None
            self.teacher_diffusion = None

        # load the target model for distillation, if path specified.

        if "_target_" in cfg.trainer.model:
            self.target_model = hydra.utils.instantiate(cfg.trainer.model)
        elif cfg.trainer.model.name == "MLP":
            assert hasattr(self, "x_dim") and hasattr(
                self, "cond_dim"
            ), "x_dim and cond_dim must be defined"
            self.target_model = MLP(
                self.x_dim, self.cond_dim, learn_sigma=cfg.trainer.model.learn_sigma
            )
        else:
            raise ValueError(f"Unknown model {cfg.trainer.model.name}")

        self.target_model.to(self.local_rank)
        self.target_model.train()

        for dst, src in zip(self.target_model.parameters(), self.model.parameters()):
            dst.data.copy_(src.data)

        if self.target_model:
            if self.resume_checkpoint:
                self.target_model.load_state_dict(self.snapshot["target_model_state"])
            self.target_model.requires_grad_(False)
            self.target_model.train()

        if self.teacher_model:
            if self.resume_checkpoint:
                self.teacher_model.load_state_dict(self.snapshot["teacher_model_state"])
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

        self.global_step = self.step
        if training_mode == "progdist":
            self.target_model.eval()
            _, scale = self.ema_scale_fn(self.global_step)
            if scale == 1 or scale == 2:
                _, start_scale = self.ema_scale_fn(0)
                n_normal_steps = int(np.log2(start_scale // 2)) * self.lr_anneal_steps
                step = self.global_step - n_normal_steps
                if step != 0:
                    self.lr_anneal_steps *= 2
                    self.step = step % self.lr_anneal_steps
                else:
                    self.step = 0
            else:
                self.step = self.global_step % self.lr_anneal_steps

    def run_loop(self):
        saved = False
        assert (
            self.total_training_steps > self.resume_step
        ), "total_training_steps must be greater than resume_step"
        for _ in trange(
            self.global_step,
            self.total_training_steps,
            initial=self.step,
            total=self.total_training_steps,
        ):
            if not (not self.lr_anneal_steps or self.step < self.lr_anneal_steps):
                break
            batch, cond = next(self.dataset)
            self.run_step(batch, cond)
            saved = False
            if (
                self.global_step
                and self.save_interval != -1
                and self.global_step % self.save_interval == 0
            ):
                self.save()
                saved = True
                th.cuda.empty_cache()

            if self.global_step % self.log_interval == 0:
                logger.dumpkvs()

        # Save the last checkpoint if it wasn't already saved.
        if self.local_rank == 0 and not saved:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_amp:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        for ema in self.ema:
            ema.update(self.model.parameters())
        if self.target_model:
            self._update_target_ema()
        if self.training_mode == "progdist":
            self.reset_training_for_progdist()
        self.step += 1
        self.global_step += 1

        self._anneal_lr()
        self.log_step()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_model.parameters(),
                self.model.parameters(),
                rate=target_ema,
            )

    def reset_training_for_progdist(self):
        assert self.training_mode == "progdist", "Training mode must be progdist"
        if self.global_step > 0:
            scales = self.ema_scale_fn(self.global_step)[1]
            scales2 = self.ema_scale_fn(self.global_step - 1)[1]
            if scales != scales2:
                with th.no_grad():
                    update_ema(
                        self.teacher_model.parameters(),
                        self.model.parameters(),
                        0.0,
                    )
                # reset optimizer
                self.opt = RAdam(
                    self.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
                self.ema = [
                    ExponentialMovingAverage(self.model.parameters(), decay=rate)
                    for rate in range(len(self.ema_rate))
                ]

                if scales == 2:
                    self.lr_anneal_steps *= 2
                self.teacher_model.eval()
                self.step = 0

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            ema, num_scales = self.ema_scale_fn(self.global_step)
            if self.training_mode == "progdist":
                if num_scales == self.ema_scale_fn(0)[1]:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.ddp_model,
                        micro,
                        num_scales,
                        target_model=self.teacher_model,
                        target_diffusion=self.teacher_diffusion,
                        model_kwargs=micro_cond,
                    )
                else:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.ddp_model,
                        micro,
                        num_scales,
                        target_model=self.target_model,
                        target_diffusion=self.diffusion,
                        model_kwargs=micro_cond,
                    )
            elif self.training_mode == "consistency_distillation":
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.ddp_model,
                    micro,
                    num_scales,
                    target_model=self.target_model,
                    teacher_model=self.teacher_model,
                    teacher_diffusion=self.teacher_diffusion,
                    model_kwargs=micro_cond,
                )
            elif self.training_mode == "consistency_training":
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.ddp_model,
                    micro,
                    num_scales,
                    target_model=self.target_model,
                    model_kwargs=micro_cond,
                )
            else:
                raise ValueError(f"Unknown training mode {self.training_mode}")
            with th.set_grad_enabled(True), th.amp.autocast(
                device_type="cuda", dtype=th.float16, enabled=self.use_amp
            ):
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

    def save(self):
        import blobfile as bf

        step = self.global_step

        snapshot = {}
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot["model_state"] = raw_model.state_dict()
        snapshot["opt_state"] = self.opt.state_dict()
        snapshot["step"] = step
        for rate, ema in zip(self.ema_rate, self.ema):
            snapshot[f"ema_{rate}"] = ema.state_dict()

        logger.log("saving optimizer state...")

        if self.target_model:
            logger.log("saving target model state")
            snapshot["target_model_state"] = self.target_model.state_dict()
        if self.teacher_model and self.training_mode == "progdist":
            logger.log("saving teacher model state")
            snapshot["teacher_model_state"] = self.teacher_model.state_dict()

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"checkpoint_{step:06d}.pt"),
            "wb",
        ) as f:
            th.save(snapshot, f)

    def log_step(self):
        step = self.global_step
        logger.logkv("step", step)
        logger.logkv("samples", (step + 1) * self.global_batch)


class SDETrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        warmup,
        grad_clip,
        continuous,
        likelihood_weighting,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.warmup = warmup
        self.grad_clip = grad_clip
        self.continuous = continuous
        self.likelihood_weighting = likelihood_weighting

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.warmup > 0:
            for g in self.opt.param_groups:
                g["lr"] = self.lr * np.minimum(self.resume_step / self.warmup, 1.0)
        if self.grad_clip >= 0:
            th.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
        if self.use_amp:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()

        for ema in self.ema:
            ema.update(self.model.parameters())

        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]

            if self.continuous:
                t = (
                    th.rand(batch.shape[0], device=dist_util.dev())
                    * (self.diffusion.T - self.eps)
                    + self.eps
                )
                compute_losses = functools.partial(
                    self._continuous_loss, micro, t, micro_cond
                )
            else:
                assert (
                    not self.likelihood_weighting
                ), "Likelihood weighting is not supported for original SMLD/DDPM training."
                t = th.randint(
                    0, self.diffusion.N, (micro.shape[0],), device=dist_util.dev()
                )
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    model=self.ddp_model,
                    batch=micro,
                    labels=t,
                    model_kwargs=micro_cond,
                )

            with th.set_grad_enabled(True), th.amp.autocast(
                device_type="cuda", dtype=th.float16, enabled=self.use_amp
            ):
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

            loss = (losses["loss"]).mean()
            log_loss_dict(self.diffusion, t, {k: v for k, v in losses.items()})

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

    def _continuous_loss(self, batch, t, model_kwargs):
        reduce_op = (
            th.mean
            if self.diffusion.reduce_mean
            else lambda *args, **kwargs: 0.5 * th.sum(*args, **kwargs)
        )

        z = th.randn_like(batch)

        mean, std = self.diffusion.marginal_prob(batch, t)
        perturbed_data = mean + _expand_tensor_shape(std, z.shape) * z
        score = self._get_score(perturbed_data, t, model_kwargs)

        if not self.likelihood_weighting:
            losses = th.square(score * _expand_tensor_shape(std, z.shape) + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = self.diffusion.sde(th.zeros_like(batch), t)[1] ** 2
            losses = th.square(score + z / _expand_tensor_shape(std, score.shape))
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        return {"loss": losses}

    def _get_score(self, x, t, model_kwargs=None):
        if isinstance(self.diffusion, VPSDE) or isinstance(self.diffusion, subVPSDE):
            if self.continuous or isinstance(self.diffusion, subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = self.ddp_model(x, labels, **model_kwargs)
                std = self.diffusion.marginal_prob(th.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (self.diffusion.N - 1)
                score = self.ddp_model(x, labels, **model_kwargs)
                std = self.diffusion.sqrt_1m_alphas_cumprod.to(labels.device)[
                    labels.long()
                ]

            score = -score / _expand_tensor_shape(std, score.shape)
            return score

        elif isinstance(self.diffusion, VESDE):
            if self.continuous:
                labels = self.diffusion.marginal_prob(th.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = self.diffusion.T - t
                labels *= self.diffusion.N - 1
                labels = th.round(labels).long()

            score = self.ddp_model(x, labels, **model_kwargs)
            return score

        else:
            raise NotImplementedError(
                f"SDE class {self.diffusion.__class__.__name__} not yet supported."
            )
