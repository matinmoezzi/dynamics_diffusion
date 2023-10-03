"""
Train a diffusion model on dynamics model.
"""

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path


from dynamics_diffusion import dist_util, logger


def sde_continuous_solver(node):
    if hasattr(node, "continuous"):
        if node.continuous:
            return "cont"
    return ""


OmegaConf.register_new_resolver("sde_continuous_solver", sde_continuous_solver)


@hydra.main(
    version_base=None,
    config_path="../config/dynamics_config",
    config_name="train_dynamics.yaml",
)
def main(cfg: DictConfig):
    log_dir = Path(HydraConfig.get().run.dir, "train").resolve()
    dist_util.setup_dist()
    logger.configure(dir=str(log_dir), format_strs=cfg.format_strs)

    hydra.utils.instantiate(cfg.trainer, cfg=cfg).run()


if __name__ == "__main__":
    main()
