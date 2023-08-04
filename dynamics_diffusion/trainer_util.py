"""
Trainer class for training the model and diffusion.
Trainer options are specified in the config file: DDPM, CM, and SDE
"""

import hydra

from dynamics_diffusion import dist_util, logger
from dynamics_diffusion.resample import create_named_schedule_sampler
from dynamics_diffusion.script_util import create_ema_and_scales_fn
from dynamics_diffusion.train_util import CMTrainLoop, TrainLoop
import torch.distributed as dist
from dynamics_diffusion.rl_datasets import get_d4rl_dataset


class Trainer:
    def __init__(self, cfg, model, diffusion, global_batch_size, **kwargs) -> None:
        self.model_cfg = model
        self.diffusion_cfg = diffusion
        self.kwargs = kwargs

        logger.log("creating data loader...")
        if kwargs["batch_size"] == -1:
            kwargs["batch_size"] = global_batch_size // dist.get_world_size()
            if global_batch_size % dist.get_world_size() != 0:
                logger.log(
                    f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {global_batch_size}"
                )

        self.data, self.info = get_d4rl_dataset(
            cfg.env.name,
            kwargs["batch_size"],
            cfg.env.deterministic_loader,
            cfg.env.reward_tune,
        )

        self.state_dim = self.info["state_dim"]
        self.action_dim = self.info["action_dim"]
        self.cond_dim = self.state_dim + self.action_dim

        logger.log("creating model and diffusion...")
        self.create_model(
            self.model_cfg, state_dim=self.state_dim, cond_dim=self.cond_dim
        )
        self.create_diffusion(self.diffusion_cfg)

    def create_model(self, model_cfg, state_dim, cond_dim):
        raise NotImplementedError

    def create_diffusion(self, diffusion_cfg):
        raise NotImplementedError

    def run(self):
        logger.log("training...")
        self._run()

    def _run(self):
        raise NotImplementedError


class DDPMTrainer(Trainer):
    def __init__(
        self, cfg, model, diffusion, global_batch_size, schedule_sampler, **kwargs
    ) -> None:
        super().__init__(cfg, model, diffusion, global_batch_size, **kwargs)
        self.schedule_sampler = create_named_schedule_sampler(
            schedule_sampler, self.diffusion
        )

    def create_model(self, model_cfg, state_dim, cond_dim):
        self.model = hydra.utils.instantiate(model_cfg.target, state_dim, cond_dim)
        self.model.to(dist_util.dev())

    def create_diffusion(self, diffusion_cfg):
        self.diffusion = hydra.utils.call(diffusion_cfg.target)

    def _run(self):
        logger.log("training...")
        TrainLoop(
            model=self.model,
            diffusion=self.diffusion,
            data=self.data,
            schedule_sampler=self.schedule_sampler,
            **self.kwargs,
        ).run_loop()


class CMTrainer(Trainer):
    def __init__(
        self,
        cfg,
        model,
        diffusion,
        global_batch_size,
        schedule_sampler,
        target_ema_mode,
        start_ema,
        scale_mode,
        start_scales,
        end_scales,
        distill_steps_per_iter,
        teacher_model_path,
        teacher_dropout,
        **kwargs,
    ) -> None:
        super().__init__(cfg, model, diffusion, global_batch_size, **kwargs)
        if kwargs["use_fp16"]:
            self.model.convert_to_fp16()

        self.schedule_sampler = create_named_schedule_sampler(
            schedule_sampler, self.diffusion
        )

        self.ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=target_ema_mode,
            start_ema=start_ema,
            scale_mode=scale_mode,
            start_scales=start_scales,
            end_scales=end_scales,
            total_steps=kwargs["total_training_steps"],
            distill_steps_per_iter=distill_steps_per_iter,
        )

        if kwargs["training_mode"] == "progdist":
            self.distillation = False
        elif "consistency" in kwargs["training_mode"]:
            self.distillation = True
        else:
            raise ValueError(f"unknown training mode {kwargs['training_mode']}")

        if len(teacher_model_path) > 0:  # path to the teacher score model.
            logger.log(f"loading the teacher model from {teacher_model_path}")
            self.teacher_diffusion = hydra.utils.instantiate(
                self.diffusion_cfg.target, distillation=self.distillation
            )
            self.teacher_model = hydra.utils.instantiate(
                self.model_cfg.target,
                x_dim=self.state_dim,
                cond_dim=self.cond_dim,
                dropout=teacher_dropout,
            )

            self.teacher_model.load_state_dict(
                dist_util.load_state_dict(teacher_model_path, map_location="cpu"),
            )

            self.teacher_model.to(dist_util.dev())
            self.teacher_model.eval()

            for dst, src in zip(
                self.model.parameters(), self.teacher_model.parameters()
            ):
                dst.data.copy_(src.data)

            if kwargs["use_fp16"]:
                self.teacher_model.convert_to_fp16()

        else:
            self.teacher_model = None
            self.teacher_diffusion = None

        logger.log("creating the target model")
        self.target_model = hydra.utils.instantiate(
            self.model_cfg.target, x_dim=self.state_dim, cond_dim=self.cond_dim
        )

        self.target_model.to(dist_util.dev())
        self.target_model.train()

        dist_util.sync_params(self.target_model.parameters())
        dist_util.sync_params(self.target_model.buffers())

        for dst, src in zip(self.target_model.parameters(), self.model.parameters()):
            dst.data.copy_(src.data)

        if kwargs["use_fp16"]:
            self.target_model.convert_to_fp16()

    def create_model(self, model_cfg, state_dim, cond_dim):
        self.model = hydra.utils.instantiate(model_cfg.target, state_dim, cond_dim)
        self.model.to(dist_util.dev())
        self.model.train()

    def create_diffusion(self, diffusion_cfg):
        self.diffusion = hydra.utils.call(diffusion_cfg.target)

    def _run(self):
        CMTrainLoop(
            model=self.model,
            diffusion=self.diffusion,
            data=self.data,
            teacher_model=self.teacher_model,
            teacher_diffusion=self.teacher_diffusion,
            target_model=self.target_model,
            schedule_sampler=self.schedule_sampler,
            ema_scale_fn=self.ema_scale_fn,
            **self.kwargs,
        ).run_loop()


class SDETrainer(Trainer):
    def __init__(self, cfg, score_model, sde, global_batch_size, **kwargs) -> None:
        super().__init__(cfg, score_model, sde, global_batch_size, **kwargs)

    def create_model(self, model_cfg, state_dim, cond_dim):
        self.model = hydra.utils.instantiate(model_cfg.target, state_dim, cond_dim)
        self.model.to(dist_util.dev())
        self.model.train()

    def create_diffusion(self, diffusion_cfg):
        self.diffusion = hydra.utils.call(diffusion_cfg.target)
