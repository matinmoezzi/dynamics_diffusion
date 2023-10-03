import os
from functools import partial
from pathlib import Path
import pathlib
import time
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra
import numpy as np
import torch as th
import torch.distributed as dist
from tqdm import trange
import yaml

from dynamics_diffusion import dist_util, logger, sde_sampling
from dynamics_diffusion.karras_diffusion import karras_sample
from dynamics_diffusion.random_util import get_generator
from dynamics_diffusion.rl_datasets import get_env
from dynamics_diffusion.script_util import random_rollout
from dynamics_diffusion.sde import VESDE


def load_yaml_file(file_path: str):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


OmegaConf.register_new_resolver("load_yaml", load_yaml_file)


@hydra.main(
    version_base=None,
    config_path="../config/dynamics_config",
    config_name="sample_dynamics",
)
def main(cfg: DictConfig):
    assert Path(cfg.model_dir, ".hydra").is_dir(), "Hydra configuration not found."

    model_prefix = "ema" if cfg.use_ema else "model"

    list_models = list(pathlib.Path(cfg.model_dir, "train").glob(f"{model_prefix}*.pt"))
    assert list_models, f"No {model_prefix} found."

    model_path = max(list_models, key=os.path.getctime)

    train_cfg = OmegaConf.load(Path(cfg.model_dir, ".hydra", "config.yaml"))

    dist_util.setup_dist()
    log_dir = Path(HydraConfig.get().run.dir).resolve()
    logger.configure(dir=str(log_dir), format_strs=["stdout"])

    env = get_env(train_cfg.env.name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    cond_dim = state_dim + action_dim

    logger.log("creating model and diffusion...")
    model = hydra.utils.instantiate(train_cfg.trainer.model.target, state_dim, cond_dim)
    model.load_state_dict(
        dist_util.load_state_dict(str(model_path), map_location="cpu")
    )
    model.to(dist_util.dev())
    if train_cfg.trainer.use_fp16:
        model.convert_to_fp16()
    model.eval()

    is_sde = False

    if train_cfg.trainer._target_.split(".")[-1] == "DDPMTrainer":
        diffusion = hydra.utils.call(train_cfg.trainer.diffusion.target)
        sample_fn = (
            diffusion.p_sample_loop if not cfg.use_ddim else diffusion.ddim_sample_loop
        )

        sample_fn_wrapper = partial(
            sample_fn,
            model,
            (cfg.batch_size, state_dim),
            clip_denoised=cfg.clip_denoised,
        )
    elif train_cfg.trainer._target_.split(".")[-1] == "CMTrainer":
        if "consistency" in train_cfg.trainer.training_mode:
            distillation = True
        else:
            distillation = False
        diffusion = hydra.utils.call(
            train_cfg.trainer.diffusion.target, distillation=distillation
        )
        if cfg.cm_sampler.sampler == "multistep":
            assert len(cfg.ts) > 0
            ts = tuple(int(x) for x in cfg.ts.split(","))
        else:
            ts = None
        generator = get_generator(
            cfg.cm_sampler.generator, cfg.num_samples, cfg.cm_sampler.seed
        )
        sample_fn_wrapper = partial(
            karras_sample,
            diffusion,
            model,
            (cfg.batch_size, state_dim),
            steps=cfg.cm_sampler.steps,
            device=dist_util.dev(),
            clip_denoised=cfg.clip_denoised,
            sampler=cfg.cm_sampler.sampler,
            sigma_min=train_cfg.trainer.diffusion.target.sigma_min,
            sigma_max=train_cfg.trainer.diffusion.target.sigma_max,
            s_churn=cfg.cm_sampler.s_churn,
            s_tmin=cfg.cm_sampler.s_tmin,
            s_tmax=cfg.cm_sampler.s_tmax,
            s_noise=cfg.cm_sampler.s_noise,
            generator=generator,
            ts=ts,
        )
    elif train_cfg.trainer._target_.split(".")[-1] == "SDETrainer":
        sde = hydra.utils.instantiate(train_cfg.trainer.diffusion.target)
        sampling_eps = 1e-3
        if isinstance(sde, VESDE):
            sampling_eps = 1e-5
        sampling_shape = (cfg.batch_size, state_dim)
        inverse_scaler = lambda x: x
        sampling_fn = partial(
            sde_sampling.get_sampling_fn,
            cfg.sde_sampler,
            sde,
            sampling_shape,
            inverse_scaler,
            sampling_eps,
            continuous=train_cfg.trainer.continuous,
            device=dist_util.dev(),
        )
        is_sde = True
    else:
        raise NotImplementedError(f"{train_cfg.trainer._target_} not supported.")

    logger.log(f"sampling {train_cfg.env.name}...")

    all_samples = []
    all_states = []
    all_actions = []
    all_next_states = []
    start = time.time()
    # while len(all_samples) * cfg.batch_size < cfg.num_samples:
    for _ in trange(0, cfg.num_samples, cfg.batch_size):
        model_kwargs = {}
        traj = random_rollout(env, cfg.batch_size)
        states = th.from_numpy(np.array(traj["states"])).float().to(dist_util.dev())
        model_kwargs["state"] = states
        all_states.extend([np.array(traj["states"])])

        actions = th.from_numpy(np.array(traj["actions"])).float().to(dist_util.dev())
        model_kwargs["action"] = actions
        all_actions.extend([np.array(traj["actions"])])
        all_next_states.extend([np.array(traj["next_states"])])

        if is_sde:
            sample, n = sampling_fn(model_kwargs=model_kwargs)(model)
        else:
            sample = sample_fn_wrapper(model_kwargs=model_kwargs)

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        logger.log(f"created {len(all_samples) * cfg.batch_size} samples")
    end = time.time()

    samples_arr = np.concatenate(all_samples, axis=0)[: cfg.num_samples]
    states_arr = np.concatenate(all_states, axis=0)[: cfg.num_samples]
    actions_arr = np.concatenate(all_actions, axis=0)[: cfg.num_samples]
    next_states_arr = np.concatenate(all_next_states, axis=0)[: cfg.num_samples]

    mse = ((next_states_arr - samples_arr) ** 2).mean(axis=0)
    logger.log(f"MSE between true next_states and sampled next_states: {mse}")
    logger.log(f"{cfg.num_samples} sampled in {end - start:.4f} sec")

    if dist.get_rank() == 0:
        out_path = Path(log_dir, f"{cfg.num_samples}samples.npz")
        logger.log(f"saving to {out_path}")
        np.savez(
            out_path,
            states=states_arr,
            actions=actions_arr,
            true_next_states=next_states_arr,
            sampled_next_states=samples_arr,
        )

    dist.barrier()
    logger.log("sampling complete")


if __name__ == "__main__":
    main()
