import datetime
import torch.distributed as dist
from pathlib import Path
import hydra
from omegaconf import DictConfig

from guided_diffusion import dist_util, logger
from guided_diffusion.rl_datasets import get_d4rl_dataset
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_ema_and_scales_fn
from guided_diffusion.train_util import CMTrainLoop, TrainLoop


@hydra.main(version_base=None, config_path="../config", config_name="cm_train_config")
def main(cfg: DictConfig):
    log_dir = f'{Path().resolve()}/logs/cm_train/{cfg.env.name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")}'

    dist_util.setup_dist()
    logger.configure(dir=log_dir, format_strs=["stdout", "torch-tensorboard"])

    logger.log("creating data loader...")
    if cfg.batch_size == -1:
        batch_size = cfg.global_batch_size // dist.get_world_size()
        if cfg.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {cfg.global_batch_size}"
            )
    else:
        batch_size = cfg.batch_size
    data, info = get_d4rl_dataset(cfg.env.name, batch_size)

    state_dim = info["state_dim"]
    action_dim = info["action_dim"]
    cond_dim = state_dim + action_dim

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=cfg.target_ema_mode,
        start_ema=cfg.start_ema,
        scale_mode=cfg.scale_mode,
        start_scales=cfg.start_scales,
        end_scales=cfg.end_scales,
        total_steps=cfg.total_training_steps,
        distill_steps_per_iter=cfg.distill_steps_per_iter,
    )
    if cfg.training_mode == "progdist":
        distillation = False
    elif "consistency" in cfg.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {cfg.training_mode}")

    diffusion = hydra.utils.instantiate(cfg.diffusion, distillation=distillation)
    model = hydra.utils.instantiate(cfg.model, x_dim=state_dim, cond_dim=cond_dim)

    model.to(dist_util.dev())
    model.train()
    if cfg.use_fp16:
        model.convert_to_fp16()

    schedule_sampler = create_named_schedule_sampler(cfg.schedule_sampler, diffusion)

    if len(cfg.teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {cfg.teacher_model_path}")
        teacher_diffusion = hydra.utils.instantiate(
            cfg.diffusion, distillation=distillation
        )
        teacher_model = hydra.utils.instantiate(
            cfg.model,
            state_dim=state_dim,
            cond_dim=cond_dim,
            dropout=cfg.teacher_dropout,
        )

        teacher_model.load_state_dict(
            dist_util.load_state_dict(cfg.teacher_model_path, map_location="cpu"),
        )

        teacher_model.to(dist_util.dev())
        teacher_model.eval()

        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        if cfg.use_fp16:
            teacher_model.convert_to_fp16()

    else:
        teacher_model = None
        teacher_diffusion = None

    # load the target model for distillation, if path specified.

    logger.log("creating the target model")
    target_model = hydra.utils.instantiate(
        cfg.model, x_dim=state_dim, cond_dim=cond_dim
    )

    target_model.to(dist_util.dev())
    target_model.train()

    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    if cfg.use_fp16:
        if hasattr(target_model, "convert_to_fp16"):
            target_model.convert_to_fp16()
        else:
            target_model = target_model.half()

    logger.log("training...")
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=cfg.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=cfg.total_training_steps,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=cfg.microbatch,
        lr=cfg.lr,
        ema_rate=cfg.ema_rate,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        resume_checkpoint=cfg.resume_checkpoint,
        use_fp16=cfg.use_fp16,
        fp16_scale_growth=cfg.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.weight_decay,
        lr_anneal_steps=cfg.lr_anneal_steps,
    ).run_loop()


if __name__ == "__main__":
    main()
