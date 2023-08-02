import torch.distributed as dist
import os
from pathlib import Path
import time
import hydra

import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch as th
from hydra.core.hydra_config import HydraConfig
from tqdm import trange

from dynamics_diffusion import dist_util, logger
from dynamics_diffusion.rl_datasets import get_env
from dynamics_diffusion.script_util import random_rollout
from dynamics_diffusion.random_util import get_generator
from dynamics_diffusion.karras_diffusion import karras_sample


@hydra.main(version_base=None, config_path="../config", config_name="cm_sample_config")
def main(cfg: DictConfig):
    assert Path(cfg.model_dir, ".hydra").is_dir(), "Hydra configuration not found."

    model_prefix = "ema" if cfg.use_ema else "model"

    list_models = list(Path(cfg.model_dir, "cm-train").glob(f"{model_prefix}*.pt"))
    assert list_models, f"No {model_prefix} found."

    model_path = max(list_models, key=os.path.getctime)

    train_cfg = OmegaConf.load(Path(cfg.model_dir, ".hydra", "config.yaml"))

    dist_util.setup_dist()
    log_dir = Path(HydraConfig.get().run.dir).resolve()
    logger.configure(dir=str(log_dir), format_strs=["stdout"])

    if "consistency" in train_cfg.training_mode:
        distillation = True
    else:
        distillation = False

    env = get_env(train_cfg.env.name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    cond_dim = state_dim + action_dim

    logger.log("creating model and diffusion...")
    diffusion = hydra.utils.call(train_cfg.diffusion.target, distillation=distillation)
    model = hydra.utils.instantiate(train_cfg.model.target, state_dim, cond_dim)
    model.load_state_dict(
        dist_util.load_state_dict(str(model_path), map_location="cpu")
    )
    model.to(dist_util.dev())
    if train_cfg.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if cfg.sampler == "multistep":
        assert len(cfg.ts) > 0
        ts = tuple(int(x) for x in cfg.ts.split(","))
    else:
        ts = None

    all_samples = []
    all_states = []
    all_actions = []
    all_next_states = []
    generator = get_generator(cfg.generator, cfg.num_samples, cfg.seed)
    start = time.time()
    for _ in trange(0, cfg.num_samples, cfg.batch_size):
        model_kwargs = {}
        traj = random_rollout(env, cfg.batch_size)
        states = th.from_numpy(np.array(traj["states"])).float().to(dist_util.dev())
        model_kwargs["state"] = states
        all_states.extend([np.array(traj["states"])])

        actions = th.from_numpy(np.array(traj["actions"])).float().to(dist_util.dev())
        model_kwargs["action"] = actions

        all_actions.extend([np.array(traj["actions"])])

        sample = karras_sample(
            diffusion,
            model,
            (cfg.batch_size, state_dim),
            steps=cfg.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=cfg.clip_denoised,
            sampler=cfg.sampler,
            sigma_min=train_cfg.diffusion.target.sigma_min,
            sigma_max=train_cfg.diffusion.target.sigma_max,
            s_churn=cfg.s_churn,
            s_tmin=cfg.s_tmin,
            s_tmax=cfg.s_tmax,
            s_noise=cfg.s_noise,
            generator=generator,
            ts=ts,
        )
        all_next_states.extend([np.array(traj["next_states"])])

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
