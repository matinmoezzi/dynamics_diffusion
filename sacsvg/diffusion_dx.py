# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import functools
import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD, RMSprop, RAdam
from dynamics_diffusion import dist_util, logger, sde_sampling
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from dynamics_diffusion.karras_diffusion import karras_sample

from dynamics_diffusion.mlp import MLP
from dynamics_diffusion.nn import update_ema
from dynamics_diffusion.random_util import get_generator
from dynamics_diffusion.resample import (
    LossAwareSampler,
    UniformSampler,
    create_named_schedule_sampler,
)
from dynamics_diffusion.script_util import create_ema_and_scales_fn
from dynamics_diffusion.sde import VESDE, VPSDE, subVPSDE
from dynamics_diffusion.sde_models.ema import ExponentialMovingAverage
from dynamics_diffusion.train_util import _expand_tensor_shape
from sacsvg.fp16_util import (
    MixedPrecisionTrainer,
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)


def get_opt(optimizer_name, model_params, lr=0.001, weight_decay=0.0):
    """
    Instantiate a torch optimizer based on the optimizer name and model parameters.

    Args:
    optimizer_name (str): Name of the optimizer to use. Should be one of 'Adam', 'AdamW', 'RAdam', 'SGD', 'RMSprop'.
    model_params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
    lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
    torch.optim.Optimizer: An instantiated optimizer object.

    Raises:
    ValueError: If the optimizer_name is not recognized.
    """
    optimizers = {
        "Adam": Adam(model_params, lr=lr, weight_decay=weight_decay),
        "AdamW": AdamW(model_params, lr=lr, weight_decay=weight_decay),
        "RAdam": RAdam(model_params, lr=lr, weight_decay=weight_decay),
        "SGD": SGD(model_params, lr=lr, weight_decay=weight_decay),
        "RMSprop": RMSprop(model_params, lr=lr, weight_decay=weight_decay),
    }

    if optimizer_name in optimizers:
        return optimizers[optimizer_name]
    else:
        raise ValueError(
            f"Optimizer '{optimizer_name}' not recognized. Choose from 'Adam', 'AdamW', 'RAdam', 'SGD', 'RMSprop'."
        )


class DiffusionDx(nn.Module):
    def __init__(
        self,
        model,
        diffusion,
        env_name,
        obs_dim,
        action_dim,
        horizon,
        detach_xt,
        clip_grad_norm,
        lr,
        opt_name,
        use_fp16,
        ema_rate,
        fp16_scale_growth,
        weight_decay,
        abs_pos=False,
    ):
        super().__init__()

        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = dist_util.DistUtil.dev()
        self.detach_xt = detach_xt
        self.clip_grad_norm = clip_grad_norm
        self.opt_name = opt_name
        self.diffusion = diffusion
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.abs_pos = abs_pos
        self.weight_decay = weight_decay

        if isinstance(model, torch.nn.Module):
            self.model = model
        elif isinstance(model, functools.partial):
            if model.func == MLP:
                self.model = model(
                    x_dim=self.obs_dim, cond_dim=self.obs_dim + self.action_dim
                )
        else:
            raise ValueError(f"Unknown model {model}")
        self.model = self.model.to(dist_util.DistUtil.dev())
        self.model.train()
        if use_fp16:
            self.model.convert_to_fp16()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )

        self.ema = [
            ExponentialMovingAverage(self.mp_trainer.master_params, decay=rate)
            for rate in range(len(self.ema_rate))
        ]

        # Manually freeze the goal locations
        if env_name == "gym_petsReacher":
            self.freeze_dims = torch.LongTensor([7, 8, 9])
        elif env_name == "gym_petsPusher":
            self.freeze_dims = torch.LongTensor([20, 21, 22])
        else:
            self.freeze_dims = None

        if dist_util.DistUtil.device == "cpu":
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=None,
                output_device=None,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )

        elif torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.DistUtil.dev()],
                output_device=dist_util.DistUtil.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.opt = get_opt(
            self.opt_name,
            self.mp_trainer.master_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    def forward(self, x, us):
        return self.unroll(x, us)

    def unroll(self, x, us):
        raise NotImplementedError

    def to(self, device):
        super().to(device)
        self.model.to(device)
        if self.use_ddp:
            self.ddp_model.to(device)
            self.ddp_model.device = torch.device(device)
            self.model.device = torch.device(device)
            self.ddp_model.device_ids = [torch.device(device).index]
            self.model.device_ids = [torch.device(device).index]
        return self


class DDPMDx(DiffusionDx):
    def __init__(self, use_ddim, clip_denoised, schedule_sampler, progress, **kwargs):
        super().__init__(**kwargs)
        self.use_ddim = use_ddim
        self.clip_denoised = clip_denoised

        schedule_sampler = create_named_schedule_sampler(
            schedule_sampler, self.diffusion
        )
        self.schedule_sampler = schedule_sampler or UniformSampler(self.diffusion)

        self.progress = progress

    def unroll_policy(self, init_x, policy, sample=True, last_u=True, detach_xt=False):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.freeze_dims is not None:
            obs_frozen = init_x[:, self.freeze_dims]

        pred_xs = []
        us = []
        log_p_us = []
        xt = init_x
        for t in range(self.horizon - 1):
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

            if detach_xt:
                xt = xt.detach()

            diffusion = self.diffusion
            sample_fn = (
                diffusion.p_sample_loop
                if not self.use_ddim
                else diffusion.ddim_sample_loop
            )

            sample_fn_wrapper = functools.partial(
                sample_fn,
                self.ddp_model,
                (n_batch, self.obs_dim),
                clip_denoised=self.clip_denoised,
                progress=self.progress,
            )
            dx_sample = sample_fn_wrapper(model_kwargs={"state": xt, "action": ut})

            if self.abs_pos:
                xtp1 = dx_sample
            else:
                # Assume that s_{t+1} = s_t + dx_t
                xtp1 = xt + dx_sample

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen

            pred_xs.append(xtp1)
            xt = xtp1

        if last_u:
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

        us = torch.stack(us)
        log_p_us = torch.stack(log_p_us).squeeze(2)
        if self.horizon <= 1:
            pred_xs = torch.empty(0, n_batch, self.obs_dim).to(init_x.device)
        else:
            pred_xs = torch.stack(pred_xs)

        return us, log_p_us, pred_xs

    def unroll(self, x, us, detach_xt=False):
        assert x.dim() == 2
        assert us.dim() == 3
        n_batch = x.size(0)
        assert us.size(1) == n_batch

        if self.freeze_dims is not None:
            obs_frozen = x[:, self.freeze_dims]

        pred_xs = []
        xt = x
        for t in range(us.size(0)):
            ut = us[t]

            if detach_xt:
                xt = xt.detach()

            diffusion = self.diffusion
            sample_fn = (
                diffusion.p_sample_loop
                if not self.use_ddim
                else diffusion.ddim_sample_loop
            )

            dx_sample = sample_fn(
                self.ddp_model,
                (n_batch, self.obs_dim),
                clip_denoised=self.clip_denoised,
                model_kwargs={"state": xt, "action": ut},
            )

            if self.abs_pos:
                xtp1 = dx_sample
            else:
                # Assume that s_{t+1} = s_t + dx_t
                xtp1 = xt + dx_sample

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen

            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = torch.stack(pred_xs)

        return pred_xs

    def update_step(self, obs, action, reward, step):
        assert obs.dim() == 3
        T, batch_size, _ = obs.shape

        diffusion_losses = []
        for horizon in range(T - 1):
            self.mp_trainer.zero_grad()

            t, weights = self.schedule_sampler.sample(
                batch_size, dist_util.DistUtil.dev()
            )

            if self.abs_pos:
                x0 = obs[horizon + 1]
            else:
                x0 = obs[horizon + 1] - obs[horizon]

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                x0,
                t,
                model_kwargs={"state": obs[horizon], "action": action[horizon]},
            )

            if self.use_ddp:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            else:
                losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            loss = (losses["loss"] * weights).mean()

            diffusion_losses.append(losses["loss"])

            self.mp_trainer.backward(loss)

            took_step = self.mp_trainer.optimize(self.opt, step)

            if took_step:
                for ema in self.ema:
                    ema.update(self.mp_trainer.master_params)

        step_diffusion_loss = torch.stack(diffusion_losses).mean().item()

        logger.logkv_mean("train_diffusion/loss", step_diffusion_loss, step)

        return torch.stack(diffusion_losses).mean().item()


class CMDx(DiffusionDx):
    def __init__(
        self,
        generator,
        sampler,
        s_churn,
        s_tmin,
        s_tmax,
        s_noise,
        steps,
        seed,
        ts,
        training_mode,
        target_ema_mode,
        start_ema,
        scale_mode,
        start_scales,
        end_scales,
        distill_steps_per_iter,
        teacher_model_path,
        lr_anneal_steps,
        schedule_sampler,
        total_training_steps,
        clip_denoised,
        num_samples,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.sampler = sampler
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.steps = steps
        self.seed = seed
        self.ts = ts
        self.num_samples = num_samples

        self.target_ema_mode = target_ema_mode
        self.start_ema = start_ema
        self.scale_mode = scale_mode
        self.start_scales = start_scales
        self.end_scales = end_scales
        self.distill_steps_per_iter = distill_steps_per_iter
        self.total_training_steps = total_training_steps

        self.training_mode = training_mode
        self.clip_denoised = clip_denoised
        self.lr_anneal_steps = lr_anneal_steps

        schedule_sampler = create_named_schedule_sampler(
            schedule_sampler, self.diffusion
        )

        self.schedule_sampler = schedule_sampler or UniformSampler(self.diffusion)

        self.ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=target_ema_mode,
            start_ema=start_ema,
            scale_mode=scale_mode,
            start_scales=start_scales,
            end_scales=end_scales,
            total_steps=total_training_steps,
            distill_steps_per_iter=distill_steps_per_iter,
        )

        if len(teacher_model_path) > 0 and os.path.exist(
            teacher_model_path
        ):  # path to the teacher score model.
            logger.log(f"loading the teacher model from {teacher_model_path}")
            self.teacher_model = copy.deepcopy(self.model)
            self.teacher_model.load_state_dict(
                torch.load(teacher_model_path, map_location=dist_util.DistUtil.dev())
            )

            self.teacher_diffusion = copy.deepcopy(self.diffusion)
            self.teacher_diffusion.distillation = False

            for dst, src in zip(
                self.model.parameters(), self.teacher_model.parameters()
            ):
                dst.data.copy_(src.data)

            self.teacher_model.to(dist_util.DistUtil.dev())
            self.teacher_model.eval()
            if self.use_fp16:
                self.teacher_model.convert_to_fp16()

        else:
            self.teacher_model = None
            self.teacher_diffusion = None

        # load the target model for distillation, if path specified.
        logger.log("creating the target model")
        if hasattr(self.model, "module"):
            self.target_model = copy.deepcopy(self.model.module)
        else:
            self.target_model = copy.deepcopy(self.model)

        self.target_model.to(dist_util.DistUtil.dev())
        self.target_model.train()

        dist_util.sync_params(self.target_model.parameters())
        dist_util.sync_params(self.target_model.buffers())

        for dst, src in zip(self.target_model.parameters(), self.model.parameters()):
            dst.data.copy_(src.data)

        if self.use_fp16:
            self.target_model.convert_to_fp16()

        if self.target_model:
            self.target_model.requires_grad_(False)
            self.target_model.train()

            self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
                self.target_model.named_parameters()
            )
            self.target_model_master_params = make_master_params(
                self.target_model_param_groups_and_shapes
            )

        if self.teacher_model:
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

        self.global_step = self.step = 0
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

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["ema_scale_fn"]
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self.ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=self.target_ema_mode,
            start_ema=self.start_ema,
            scale_mode=self.scale_mode,
            start_scales=self.start_scales,
            end_scales=self.end_scales,
            total_steps=self.total_training_steps,
            distill_steps_per_iter=self.distill_steps_per_iter,
        )

    def unroll_policy(self, init_x, policy, sample=True, last_u=True, detach_xt=False):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.freeze_dims is not None:
            obs_frozen = init_x[:, self.freeze_dims]

        pred_xs = []
        us = []
        log_p_us = []
        xt = init_x
        for t in range(self.horizon - 1):
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

            if detach_xt:
                xt = xt.detach()

            if self.sampler == "multistep":
                assert len(self.ts) > 0
                ts = tuple(int(x) for x in self.ts.split(","))
            else:
                ts = None
            generator = get_generator(self.generator, self.num_samples, self.seed)

            dx_sample = karras_sample(
                self.diffusion,
                self.model,
                (n_batch, self.obs_dim),
                steps=self.steps,
                device=dist_util.DistUtil.dev(),
                clip_denoised=self.clip_denoised,
                sampler=self.sampler,
                sigma_min=self.diffusion.sigma_min,
                sigma_max=self.diffusion.sigma_max,
                s_churn=self.s_churn,
                s_tmin=self.s_tmin,
                s_tmax=self.s_tmax,
                s_noise=self.s_noise,
                generator=generator,
                ts=ts,
                model_kwargs={"state": xt, "action": ut},
            )

            if self.abs_pos:
                xtp1 = dx_sample
            else:
                # Assume that s_{t+1} = s_t + dx_t
                xtp1 = xt + dx_sample

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen

            pred_xs.append(xtp1)
            xt = xtp1

        if last_u:
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

        us = torch.stack(us)
        log_p_us = torch.stack(log_p_us).squeeze(2)
        if self.horizon <= 1:
            pred_xs = torch.empty(0, n_batch, self.obs_dim).to(init_x.device)
        else:
            pred_xs = torch.stack(pred_xs)

        return us, log_p_us, pred_xs

    def unroll(self, x, us, detach_xt=False):
        assert x.dim() == 2
        assert us.dim() == 3
        n_batch = x.size(0)
        assert us.size(1) == n_batch

        if self.freeze_dims is not None:
            obs_frozen = x[:, self.freeze_dims]

        pred_xs = []
        xt = x
        for t in range(us.size(0)):
            ut = us[t]

            if detach_xt:
                xt = xt.detach()

            if self.sampler == "multistep":
                assert len(self.ts) > 0
                ts = tuple(int(x) for x in self.ts.split(","))
            else:
                ts = None

            generator = get_generator(self.generator, self.num_samples, self.seed)

            dx_sample = karras_sample(
                self.diffusion,
                self.model,
                (n_batch, self.obs_dim),
                steps=self.steps,
                device=dist_util.DistUtil.dev(),
                clip_denoised=self.clip_denoised,
                sampler=self.sampler,
                sigma_min=self.diffusion.sigma_min,
                sigma_max=self.diffusion.sigma_max,
                s_churn=self.s_churn,
                s_tmin=self.s_tmin,
                s_tmax=self.s_tmax,
                s_noise=self.s_noise,
                generator=generator,
                ts=ts,
                model_kwargs={"state": xt, "action": ut},
            )

            if self.abs_pos:
                xtp1 = dx_sample
            else:
                # Assume that s_{t+1} = s_t + dx_t
                xtp1 = xt + dx_sample

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen

            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = torch.stack(pred_xs)

        return pred_xs

    def update_step(self, obs, action, reward, step):
        assert obs.dim() == 3
        T, batch_size, _ = obs.shape

        diffusion_losses = []
        for horizon in range(T - 1):
            self.mp_trainer.zero_grad()
            t, weights = self.schedule_sampler.sample(
                batch_size, dist_util.DistUtil.dev()
            )

            ema, num_scales = self.ema_scale_fn(self.step)

            if self.abs_pos:
                x0 = obs[horizon + 1]
            else:
                x0 = obs[horizon + 1] - obs[horizon]

            model_kwargs = {"state": obs[horizon], "action": action[horizon]}
            if self.training_mode == "progdist":
                if num_scales == self.ema_scale_fn(0)[1]:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.ddp_model,
                        x0,
                        num_scales,
                        target_model=self.teacher_model,
                        target_diffusion=self.teacher_diffusion,
                        model_kwargs=model_kwargs,
                    )
                else:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.ddp_model,
                        x0,
                        num_scales,
                        target_model=self.target_model,
                        target_diffusion=self.diffusion,
                        model_kwargs=model_kwargs,
                    )
            elif self.training_mode == "consistency_distillation":
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.ddp_model,
                    x0,
                    num_scales,
                    target_model=self.target_model,
                    teacher_model=self.teacher_model,
                    teacher_diffusion=self.teacher_diffusion,
                    model_kwargs=model_kwargs,
                )
            elif self.training_mode == "consistency_training":
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.ddp_model,
                    x0,
                    num_scales,
                    target_model=self.target_model,
                    model_kwargs=model_kwargs,
                )
            else:
                raise ValueError(f"Unknown training mode {self.training_mode}")

            if self.use_ddp:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            else:
                losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            diffusion_losses.append(losses["loss"])

            self.mp_trainer.backward(loss)

            took_step = self.mp_trainer.optimize(self.opt, step)

            if took_step:
                for ema in self.ema:
                    ema.update(self.mp_trainer.master_params)

                if self.target_model:
                    self._update_target_ema()

                if self.training_mode == "progdist":
                    self.reset_training_for_progdist()

                self.global_step += 1
                self.step += 1

        step_diffusion_loss = torch.stack(diffusion_losses).mean().item()

        logger.logkv_mean("train_diffusion/loss", step_diffusion_loss, step)

        return torch.stack(diffusion_losses).mean().item()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with torch.no_grad():
            update_ema(
                self.target_model.parameters(),
                self.model.parameters(),
                rate=target_ema,
            )

            master_params_to_model_params(
                self.target_model_param_groups_and_shapes,
                self.target_model_master_params,
            )

    def reset_training_for_progdist(self):
        assert self.training_mode == "progdist", "Training mode must be progdist"
        if self.global_step > 0:
            scales = self.ema_scale_fn(self.global_step)[1]
            scales2 = self.ema_scale_fn(self.global_step - 1)[1]
            if scales != scales2:
                with torch.no_grad():
                    update_ema(
                        self.teacher_model.parameters(),
                        self.model.parameters(),
                        0.0,
                    )
                # reset optimizer
                self.opt = RAdam(
                    self.mp_trainer.master_params,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
                self.ema = [
                    ExponentialMovingAverage(self.mp_trainer.master_params, decay=rate)
                    for rate in range(len(self.ema_rate))
                ]
                if scales == 2:
                    self.lr_anneal_steps *= 2
                self.teacher_model.eval()
                self.step = 0


class ScoreSDEDx(DiffusionDx):
    def __init__(
        self, sde_sampler, sde_continuous, sde_likelihood_weighting, sde_eps, **kwargs
    ):
        super().__init__(**kwargs)

        self.sde_sampler = sde_sampler
        self.sde_continuous = sde_continuous
        self.sde_eps = sde_eps
        self.sde_likelihood_weighting = sde_likelihood_weighting

    def unroll_policy(self, init_x, policy, sample=True, last_u=True, detach_xt=False):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.freeze_dims is not None:
            obs_frozen = init_x[:, self.freeze_dims]

        pred_xs = []
        us = []
        log_p_us = []
        xt = init_x
        for t in range(self.horizon - 1):
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

            if detach_xt:
                xt = xt.detach()

            # State-action embedding
            # xu_emb = self.xu_enc(xut).unsqueeze(0)

            # Sampling ScoreSDEDx
            sde = self.diffusion
            sampling_eps = 1e-3
            if isinstance(sde, VESDE):
                sampling_eps = 1e-5
            sampling_shape = (n_batch, self.obs_dim)
            inverse_scaler = lambda x: x
            sampling_fn = functools.partial(
                sde_sampling.get_sampling_fn,
                self.sde_sampler,
                sde,
                sampling_shape,
                inverse_scaler,
                sampling_eps,
                continuous=self.sde_continuous,
                device=dist_util.DistUtil.dev(),
            )
            model_kwargs = {"state": xt, "action": ut}
            dx_sample, n = sampling_fn(model_kwargs=model_kwargs)(self.model)

            if self.abs_pos:
                xtp1 = dx_sample
            else:
                # Assume that s_{t+1} = s_t + dx_t
                xtp1 = xt + dx_sample

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen

            pred_xs.append(xtp1)
            xt = xtp1

        if last_u:
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

        us = torch.stack(us)
        log_p_us = torch.stack(log_p_us).squeeze(2)
        if self.horizon <= 1:
            pred_xs = torch.empty(0, n_batch, self.obs_dim).to(init_x.device)
        else:
            pred_xs = torch.stack(pred_xs)

        return us, log_p_us, pred_xs

    def unroll(self, x, us, detach_xt=False):
        assert x.dim() == 2
        assert us.dim() == 3
        n_batch = x.size(0)
        assert us.size(1) == n_batch

        if self.freeze_dims is not None:
            obs_frozen = x[:, self.freeze_dims]

        pred_xs = []
        xt = x
        for t in range(us.size(0)):
            ut = us[t]

            if detach_xt:
                xt = xt.detach()

            # State-action embedding
            # xu_emb = self.xu_enc(xut).unsqueeze(0)

            # Sampling ScoreSDEDx
            sde = self.diffusion
            sampling_eps = 1e-3
            if isinstance(sde, VESDE):
                sampling_eps = 1e-5
            sampling_shape = (n_batch, self.obs_dim)
            inverse_scaler = lambda x: x
            sampling_fn = functools.partial(
                sde_sampling.get_sampling_fn,
                self.sde_sampler,
                sde,
                sampling_shape,
                inverse_scaler,
                sampling_eps,
                continuous=self.sde_continuous,
                device=dist_util.DistUtil.dev(),
            )
            model_kwargs = {"state": xt, "action": ut}
            dx_sample, n = sampling_fn(model_kwargs=model_kwargs)(self.model)

            if self.abs_pos:
                xtp1 = dx_sample
            else:
                # Assume that s_{t+1} = s_t + dx_t
                xtp1 = xt + dx_sample

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen
            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = torch.stack(pred_xs)

        return pred_xs

    def update_step(self, obs, action, reward, step):
        assert obs.dim() == 3
        T, batch_size, _ = obs.shape

        diffusion_losses = []
        for horizon in range(T - 1):
            self.mp_trainer.zero_grad()

            model_kwargs = {"state": obs[horizon], "action": action[horizon]}

            if self.abs_pos:
                x0 = obs[horizon + 1]
            else:
                x0 = obs[horizon + 1] - obs[horizon]

            # Score SDE Score-matching Loss
            if self.sde_continuous:
                t = (
                    torch.rand(batch_size, device=dist_util.DistUtil.dev())
                    * (self.diffusion.T - self.sde_eps)
                    + self.sde_eps
                )
                compute_losses = functools.partial(
                    self._continuous_loss,
                    x0,
                    t,
                    model_kwargs,
                )
            else:
                assert (
                    not self.sde_likelihood_weighting
                ), "Likelihood weighting is not supported for original SMLD/DDPM training."
                t = torch.randint(
                    0,
                    self.diffusion.N,
                    (batch_size,),
                    device=dist_util.DistUtil.dev(),
                )
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    model=self.ddp_model,
                    batch=x0,
                    labels=t,
                    model_kwargs=model_kwargs,
                )

            if self.use_ddp:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            else:
                losses = compute_losses()

            loss = (losses["loss"]).mean()

            diffusion_losses.append(losses["loss"])

            self.mp_trainer.backward(loss)

            took_step = self.mp_trainer.optimize(self.opt, step)

            if took_step:
                for ema in self.ema:
                    ema.update(self.mp_trainer.master_params)

        step_diffusion_loss = torch.stack(diffusion_losses).mean().item()

        logger.logkv_mean("train_diffusion/loss", step_diffusion_loss, step)

        return torch.stack(diffusion_losses).mean().item()

    def _continuous_loss(self, batch, t, model_kwargs):
        reduce_op = (
            torch.mean
            if self.diffusion.reduce_mean
            else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        )

        z = torch.randn_like(batch)

        mean, std = self.diffusion.marginal_prob(batch, t)
        perturbed_data = mean + _expand_tensor_shape(std, z.shape) * z
        score = self._get_score(perturbed_data, t, model_kwargs)

        if not self.sde_likelihood_weighting:
            losses = torch.square(score * _expand_tensor_shape(std, z.shape) + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = self.diffusion.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / _expand_tensor_shape(std, score.shape))
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        return {"loss": losses}

    def _get_score(self, x, t, model_kwargs=None):
        if isinstance(self.diffusion, VPSDE) or isinstance(self.diffusion, subVPSDE):
            if self.sde_continuous or isinstance(self.diffusion, subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = self.ddp_model(x, labels, **model_kwargs)
                std = self.diffusion.marginal_prob(torch.zeros_like(x), t)[1]
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
            if self.sde_continuous:
                labels = self.diffusion.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = self.diffusion.T - t
                labels *= self.diffusion.N - 1
                labels = torch.round(labels).long()

            score = self.ddp_model(x, labels, **model_kwargs)
            return score

        else:
            raise NotImplementedError(
                f"SDE class {self.diffusion.__class__.__name__} not yet supported."
            )
