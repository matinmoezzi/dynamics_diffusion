"""
Train a diffusion model on dynamics model.
"""

from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path


from dynamics_diffusion import dist_util, logger


@hydra.main(version_base=None, config_path="../config", config_name="train_config.yaml")
def main(cfg: DictConfig):
    log_dir = Path(HydraConfig.get().run.dir, "train").resolve()
    dist_util.setup_dist()
    logger.configure(dir=str(log_dir), format_strs=cfg.format_strs)

    hydra.utils.instantiate(cfg.trainer, cfg=cfg).run()


if __name__ == "__main__":
    main()
