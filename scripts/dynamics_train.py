"""
Train a diffusion model on dynamics model.
"""

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path


from dynamics_diffusion import dist_util, logger
from dynamics_diffusion.script_util import ConfigStore


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


@hydra.main(
    version_base=None,
    config_path="../config/dynamics",
    config_name="train.yaml",
)
def main(cfg: DictConfig):
    ConfigStore.set_config(cfg)

    log_dir = Path(HydraConfig.get().run.dir, "train").resolve()
    dist_util.setup_dist()
    logger.configure(dir=str(log_dir), format_strs=cfg.format_strs)

    trainer = hydra.utils.instantiate(cfg.trainer)

    logger.log("training...")
    trainer.run_loop()


if __name__ == "__main__":
    main()
