"""
Train a diffusion model on dynamics model.
"""

import datetime
from omegaconf import DictConfig
import hydra
from pathlib import Path

from dynamics_diffusion import dist_util, logger
from dynamics_diffusion.rl_datasets import get_d4rl_dataset
from dynamics_diffusion.resample import create_named_schedule_sampler
from dynamics_diffusion.train_util import TrainLoop


@hydra.main(version_base=None, config_path="../config", config_name="train_config")
def main(cfg: DictConfig):
    log_dir = f'{Path().resolve()}/logs/train/{cfg.env.name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")}'
    dist_util.setup_dist()
    logger.configure(dir=log_dir, format_strs=["stdout", "torch-tensorboard"])

    logger.log("creating data loader...")
    data, info = get_d4rl_dataset(
        cfg.env.name, cfg.batch_size, cfg.env.deterministic_loader, cfg.env.reward_tune
    )

    state_dim = info["state_dim"]
    action_dim = info["action_dim"]
    cond_dim = state_dim + action_dim

    if cfg.training_iter == -1:
        training_iter = int(info["size"])
    else:
        training_iter = int(min(cfg.training_iter, info["size"]))

    logger.log("creating model and diffusion...")
    diffusion = hydra.utils.call(cfg.diffusion)
    model = hydra.utils.instantiate(cfg.model, state_dim, cond_dim)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(cfg.schedule_sampler, diffusion)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        training_iter=training_iter,
        batch_size=cfg.batch_size,
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
