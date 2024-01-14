"""
Train a diffusion model on dynamics model.
"""

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import wandb


from dynamics_diffusion import dist_util, logger
from dynamics_diffusion.script_util import ConfigStore


def steps_to_human_readable(step_count) -> str:
    # Ensure the input is an integer
    step_count = int(step_count)

    # Convert to human-readable format
    if step_count < 1000:
        return str(step_count)
    elif step_count < 1000000:
        return f"{step_count/1000:.0f}K"  # for thousands
    else:
        return f"{step_count/1000000:.0f}M"  # for millions


def sde_continuous_solver(node):
    if hasattr(node, "continuous"):
        if node.continuous:
            return "cont"
    return ""


def get_runtime_choice(key):
    instance = HydraConfig.get()
    return instance.runtime.choices[f"{key}@trainer.{key}"]


def karras_distillation(training_mode):
    if training_mode == "progdist":
        return False
    elif "consistency" in training_mode:
        return True
    else:
        raise ValueError(f"Unknown training mode {training_mode}")


OmegaConf.register_new_resolver("sde_continuous_solver", sde_continuous_solver)
OmegaConf.register_new_resolver("get_runtime_choice", get_runtime_choice)
OmegaConf.register_new_resolver("karras_distillation", karras_distillation)
OmegaConf.register_new_resolver("human_readable_steps", steps_to_human_readable)


@hydra.main(
    version_base=None,
    config_path="../config/dynamics",
    config_name="train.yaml",
)
def main(cfg: DictConfig):
    ConfigStore.set_config(cfg)

    hydra_cfg = HydraConfig.get()

    wandb.init(
        project="diffusion_offline_RL",
        sync_tensorboard=True,
        config=OmegaConf.to_container(cfg),
        name=f"{hydra_cfg.runtime.choices['dataset@trainer.dataset']}-{hydra_cfg.runtime.choices['trainer']}-{hydra_cfg.runtime.choices['diffusion@trainer.diffusion']}-{hydra_cfg.runtime.choices['model@trainer.model']}-{steps_to_human_readable(cfg.trainer.total_training_steps)}",
    )

    log_dir = Path(hydra_cfg.run.dir, "train").resolve()
    dist_util.DistUtil.setup_dist(device=cfg.device)
    log_suffix = f"[{dist_util.DistUtil.device}]"
    logger.configure(
        dir=str(log_dir), format_strs=cfg.format_strs, log_suffix=log_suffix
    )

    trainer = hydra.utils.instantiate(cfg.trainer)

    logger.log("training...")
    trainer.run_loop()


if __name__ == "__main__":
    main()
