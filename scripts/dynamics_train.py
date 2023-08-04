"""
Train a diffusion model on dynamics model.
"""

from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path


from dynamics_diffusion import dist_util, logger
from dynamics_diffusion.train_util import SDETrainLoop


def sde_train(score_model, sde, data, data_info, **kwargs):
    score_model_cfg = score_model

    state_dim = data_info["state_dim"]
    action_dim = data_info["action_dim"]
    cond_dim = state_dim + action_dim

    logger.log("creating score model and sde...")
    sde = hydra.utils.instantiate(sde.target)
    score_model = hydra.utils.instantiate(score_model_cfg.target, state_dim, cond_dim)
    score_model.to(dist_util.dev())

    logger.log("training...")
    SDETrainLoop(score_model=score_model, sde=sde, data=data, **kwargs).run_loop()
    pass


@hydra.main(version_base=None, config_path="../config", config_name="train_config.yaml")
def main(cfg: DictConfig):
    log_dir = Path(HydraConfig.get().run.dir, "train").resolve()
    dist_util.setup_dist()
    logger.configure(dir=str(log_dir), format_strs=cfg.format_strs)

    hydra.utils.instantiate(cfg.trainer, cfg=cfg).run()


if __name__ == "__main__":
    main()
