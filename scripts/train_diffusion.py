"""
Train a diffusion model on dynamics model.
"""

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path


from dynamics_diffusion import dist_util, logger
from dynamics_diffusion.script_util import ConfigStore
from dynamics_diffusion.utils import (
    get_runtime_choice,
    karras_distillation,
    sde_continuous_solver,
    steps_to_human_readable,
)


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

    log_dir = Path(HydraConfig.get().run.dir, "train").resolve()
    dist_util.DistUtil.setup_dist(device=cfg.device)
    log_suffix = (
        f"[{dist_util.DistUtil.device.upper()}:{dist_util.DistUtil.get_global_rank()}]"
    )
    logger.configure(
        dir=str(log_dir), format_strs=cfg.format_strs, log_suffix=log_suffix
    )
    logger.log(f"Configuration:\n{cfg}")

    trainer = hydra.utils.instantiate(cfg.trainer)

    logger.log("training...")
    trainer.run_loop()


if __name__ == "__main__":
    main()
