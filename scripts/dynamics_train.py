"""
Train a diffusion model on dynamics model.
"""

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import torch.distributed as dist

from dynamics_diffusion import dist_util, logger
from dynamics_diffusion.rl_datasets import get_d4rl_dataset
from dynamics_diffusion.resample import create_named_schedule_sampler
from dynamics_diffusion.script_util import create_ema_and_scales_fn
from dynamics_diffusion.train_util import CMTrainLoop, TrainLoop


def ddpm_train(
    model,
    diffusion,
    data,
    data_info,
    training_iter,
    batch_size,
    microbatch,
    lr,
    ema_rate,
    log_interval,
    save_interval,
    resume_checkpoint,
    use_fp16,
    fp16_scale_growth,
    schedule_sampler,
    weight_decay,
    lr_anneal_steps,
):
    state_dim = data_info["state_dim"]
    action_dim = data_info["action_dim"]
    cond_dim = state_dim + action_dim

    if training_iter == -1:
        training_iter = int(data_info["size"])
    else:
        training_iter = int(min(training_iter, data_info["size"]))

    logger.log("creating model and diffusion...")
    diffusion = hydra.utils.call(diffusion.target)
    model = hydra.utils.instantiate(model.target, state_dim, cond_dim)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(schedule_sampler, diffusion)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        training_iter=training_iter,
        batch_size=batch_size,
        microbatch=microbatch,
        lr=lr,
        ema_rate=ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=use_fp16,
        fp16_scale_growth=fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=weight_decay,
        lr_anneal_steps=lr_anneal_steps,
    ).run_loop()


def cm_train(
    model,
    diffusion,
    data,
    data_info,
    total_training_steps,
    batch_size,
    microbatch,
    lr,
    ema_rate,
    log_interval,
    save_interval,
    resume_checkpoint,
    use_fp16,
    fp16_scale_growth,
    schedule_sampler,
    weight_decay,
    lr_anneal_steps,
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    training_mode,
    teacher_model_path,
    teacher_dropout,
    distill_steps_per_iter,
):
    state_dim = data_info["state_dim"]
    action_dim = data_info["action_dim"]
    cond_dim = state_dim + action_dim

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=target_ema_mode,
        start_ema=start_ema,
        scale_mode=scale_mode,
        start_scales=start_scales,
        end_scales=end_scales,
        total_steps=total_training_steps,
        distill_steps_per_iter=distill_steps_per_iter,
    )
    model_cfg = model
    diffusion_cfg = diffusion
    if training_mode == "progdist":
        distillation = False
    elif "consistency" in training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {training_mode}")

    diffusion = hydra.utils.instantiate(diffusion_cfg.target, distillation=distillation)
    model = hydra.utils.instantiate(
        model_cfg.target, x_dim=state_dim, cond_dim=cond_dim
    )

    model.to(dist_util.dev())
    model.train()
    if use_fp16:
        model.convert_to_fp16()

    schedule_sampler = create_named_schedule_sampler(schedule_sampler, diffusion)

    if len(teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {teacher_model_path}")
        teacher_diffusion = hydra.utils.instantiate(
            diffusion.target, distillation=distillation
        )
        teacher_model = hydra.utils.instantiate(
            model_cfg.target,
            x_dim=state_dim,
            cond_dim=cond_dim,
            dropout=teacher_dropout,
        )

        teacher_model.load_state_dict(
            dist_util.load_state_dict(teacher_model_path, map_location="cpu"),
        )

        teacher_model.to(dist_util.dev())
        teacher_model.eval()

        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        if use_fp16:
            teacher_model.convert_to_fp16()

    else:
        teacher_model = None
        teacher_diffusion = None

    # load the target model for distillation, if path specified.

    logger.log("creating the target model")
    target_model = hydra.utils.instantiate(
        model_cfg.target, x_dim=state_dim, cond_dim=cond_dim
    )

    target_model.to(dist_util.dev())
    target_model.train()

    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    if use_fp16:
        if hasattr(target_model, "convert_to_fp16"):
            target_model.convert_to_fp16()
        else:
            target_model = target_model.half()

    if total_training_steps == -1:
        total_training_steps = int(data_info["size"])
    else:
        total_training_steps = int(min(total_training_steps, data_info["size"]))

    logger.log("training...")
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=total_training_steps,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=microbatch,
        lr=lr,
        ema_rate=ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=use_fp16,
        fp16_scale_growth=fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=weight_decay,
        lr_anneal_steps=lr_anneal_steps,
    ).run_loop()
    pass


def sde_train():
    pass


@hydra.main(version_base=None, config_path="../config", config_name="train_config.yaml")
def main(cfg: DictConfig):
    log_dir = Path(HydraConfig.get().run.dir, "train").resolve()
    dist_util.setup_dist()
    logger.configure(dir=str(log_dir), format_strs=cfg.format_strs)
    logger.log("creating data loader...")
    if cfg.trainer.batch_size == -1:
        batch_size = cfg.global_batch_size // dist.get_world_size()
        if cfg.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {cfg.global_batch_size}"
            )
    else:
        batch_size = cfg.trainer.batch_size
    data, info = get_d4rl_dataset(
        cfg.env.name,
        batch_size,
        cfg.env.deterministic_loader,
        cfg.env.reward_tune,
    )

    hydra.utils.call(cfg.trainer, data=data, data_info=info, batch_size=batch_size)


if __name__ == "__main__":
    main()
