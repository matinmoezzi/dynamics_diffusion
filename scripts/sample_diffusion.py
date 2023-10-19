import copy
import os
from pathlib import Path
import pathlib
from omegaconf import DictConfig, OmegaConf, open_dict, read_write
from hydra import initialize, compose
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import _save_config, configure_log

from dynamics_diffusion import dist_util, logger

from scripts.train_diffusion import (
    karras_distillation,
    sde_continuous_solver,
    steps_to_human_readable,
)

NUM_CLASSES = 1000

import re


def del_key(cfg: DictConfig, key: str):
    with read_write(cfg):
        with open_dict(cfg):
            del cfg[key]


def get_runtime_choice(key):
    instance = HydraConfig.get()
    return instance.runtime.choices[f"{key}@sampler.{key}"]


def extract_step_number(path: str):
    # Extract step number using regex from the path
    match = re.search(r"checkpoint_(\d+).pt", path)
    if match:
        return str(int(match.group(1)))
    return 0  # Return -1 or any default value if pattern not found


OmegaConf.register_new_resolver(
    "extract_step_number", extract_step_number, replace=True
)
OmegaConf.register_new_resolver("get_runtime_choice", get_runtime_choice, replace=True)
OmegaConf.register_new_resolver(
    "karras_distillation", karras_distillation, replace=True
)
OmegaConf.register_new_resolver(
    "sde_continuous_solver", sde_continuous_solver, replace=True
)
OmegaConf.register_new_resolver(
    "human_readable_steps", steps_to_human_readable, replace=True
)


def main():
    old_cwd = os.getcwd()
    sample_cfg = OmegaConf.load(Path(os.getcwd(), "config/dynamics/sample.yaml"))
    assert sample_cfg.model_dir, "Model directory should be provided."
    if sample_cfg.model_dir.split(".")[-1] in ["pt", "pth"]:
        assert os.path.isfile(
            sample_cfg.model_dir
        ), f"Model {sample_cfg.model_dir} not found."
        initialize(
            version_base=None,
            config_path="../config/dynamics",
            job_name="sample_diffusion",
        )
        cfg = compose(config_name="sample.yaml", return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        output_dir = str(OmegaConf.select(cfg, "hydra.run.dir"))
        Path(str(output_dir)).mkdir(parents=True, exist_ok=True)
        hydra_cfg = HydraConfig.instance().cfg
        task_cfg = copy.deepcopy(cfg)
        model_cfg = copy.deepcopy(cfg.sampler.model)
        diffusion_cfg = copy.deepcopy(cfg.sampler.diffusion)
        dataset_cfg = copy.deepcopy(cfg.sampler.dataset)
        del_key(task_cfg, "hydra")
        with read_write(cfg.hydra.runtime):
            with open_dict(cfg.hydra.runtime):
                cfg.hydra.runtime.output_dir = os.path.abspath(output_dir)
        os.chdir(output_dir)
        configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
        _save_config(task_cfg, "config.yaml", Path(".hydra"))
        _save_config(hydra_cfg, "hydra.yaml", Path(".hydra"))
        _save_config(cfg.hydra.overrides.task, "overrides.yaml", Path(".hydra"))
        log_dir = sample_dir = Path(output_dir, "sample")
        model_checkpoint = cfg.model_dir
        os.chdir(old_cwd)
    else:
        assert os.path.isdir(
            sample_cfg.model_dir
        ), f"Model {sample_cfg.model_dir} not found."
        assert Path(
            sample_cfg.model_dir, ".hydra"
        ).is_dir(), "Hydra configuration not found."
        hydra_cfg = OmegaConf.load(Path(sample_cfg.model_dir, ".hydra", "hydra.yaml"))

        list_models = list(pathlib.Path(sample_cfg.model_dir, "train").glob(f"*.pt"))
        assert list_models, f"No model found."

        model_checkpoint = str(max(list_models, key=os.path.getctime))

        sampler = hydra_cfg.hydra.runtime.choices.trainer
        initialize(config_path="../config/dynamics", job_name="sample_diffusion")
        cfg = compose(
            config_name="sample.yaml",
            return_hydra_config=True,
            overrides=[f"sampler={sampler}"],
        )
        task_cfg = OmegaConf.load(Path(cfg.model_dir, ".hydra", "config.yaml"))
        model_cfg = task_cfg.trainer.model
        diffusion_cfg = task_cfg.trainer.diffusion
        dataset_cfg = task_cfg.trainer.dataset
        OmegaConf.update(cfg, "sampler.model", model_cfg)
        OmegaConf.update(cfg, "sampler.diffusion", diffusion_cfg)
        OmegaConf.update(cfg, "sampler.dataset", dataset_cfg)

        task_cfg = copy.deepcopy(cfg)
        OmegaConf.set_struct(task_cfg, False)
        del task_cfg["hydra"]
        OmegaConf.set_struct(task_cfg, True)

        sample_dir = log_dir = Path(cfg.model_dir, "sample")

    dist_util.DistUtil.setup_dist(sample_cfg.device)
    logger.configure(dir=str(log_dir), format_strs=["stdout"])

    logger.log("creating sampler...")

    del_key(model_cfg, "_target_")
    del_key(diffusion_cfg, "_target_")
    del_key(dataset_cfg, "_target_")

    sampler = hydra.utils.instantiate(
        task_cfg.sampler,
        model_cfg=model_cfg,
        diffusion_cfg=diffusion_cfg,
        dataset_cfg=dataset_cfg,
        model_checkpoint=model_checkpoint,
        sample_dir=str(sample_dir),
    )
    sampler.sample(sample_cfg.save)


if __name__ == "__main__":
    main()
