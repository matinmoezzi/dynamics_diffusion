import functools
import os
from pathlib import Path
import time
import numpy as np
import torch
from tqdm import trange
from dynamics_diffusion import dist_util, logger, sde_sampling
from dynamics_diffusion.karras_diffusion import karras_sample
from dynamics_diffusion.mlp import MLP
from dynamics_diffusion.random_util import get_generator
from dynamics_diffusion.rl_datasets import get_env
import torch.distributed as dist

from dynamics_diffusion.script_util import random_rollout
from dynamics_diffusion.sde import VESDE


NUM_CLASSES = 1000


class Sampler:
    def __init__(
        self,
        model,
        diffusion,
        dataset,
        model_checkpoint,
        num_samples,
        batch_size,
        sample_dir,
        model_cfg,
        diffusion_cfg,
        dataset_cfg,
        use_fp16,
    ):
        self.model = model
        self.diffusion = diffusion
        self.dataset = dataset
        self.model_checkpoint = model_checkpoint
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.sample_dir = sample_dir
        self.model_cfg = model_cfg
        self.diffusion_cfg = diffusion_cfg
        self.dataset_cfg = dataset_cfg

        if isinstance(model, torch.nn.Module):
            self.model = model
        elif isinstance(model, functools.partial):
            if model.func == MLP:
                self.env = get_env(self.dataset_cfg.name)
                self.state_dim = self.env.observation_space.shape[0]
                self.action_dim = self.env.action_space.shape[0]
                self.cond_dim = self.state_dim + self.action_dim
                self.model = model(x_dim=self.state_dim, cond_dim=self.cond_dim)
        else:
            raise ValueError(f"Unknown model {model}")

        self.model.to(dist_util.DistUtil.dev())

        checkpoint_dict = torch.load(
            self.model_checkpoint, map_location=dist_util.DistUtil.dev()
        )
        if checkpoint_dict.get("model_state", None):
            self.model.load_state_dict(checkpoint_dict["model_state"])
        else:
            self.model.load_state_dict(checkpoint_dict)

    def _get_sample_fn_dynamics(self):
        raise NotImplementedError

    def _get_sample_fn_image(self):
        raise NotImplementedError

    def _sample_dynamics(self, sampling_fn, save):
        all_samples = []
        all_states = []
        all_actions = []
        all_next_states = []
        start = time.time()
        for _ in trange(0, self.num_samples, self.batch_size):
            model_kwargs = {}
            traj = random_rollout(self.env, self.batch_size)
            states = (
                torch.from_numpy(np.array(traj["states"]))
                .float()
                .to(dist_util.DistUtil.dev())
            )
            model_kwargs["state"] = states
            all_states.extend([np.array(traj["states"])])

            actions = (
                torch.from_numpy(np.array(traj["actions"]))
                .float()
                .to(dist_util.DistUtil.dev())
            )
            model_kwargs["action"] = actions
            all_actions.extend([np.array(traj["actions"])])
            all_next_states.extend([np.array(traj["next_states"])])

            if isinstance(self, SDESampler):
                sample, n = sampling_fn(model_kwargs=model_kwargs)(self.model)
            else:
                sample = sampling_fn(model_kwargs=model_kwargs)

            gathered_samples = [
                torch.zeros_like(sample) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

            logger.log(f"created {len(all_samples) * self.batch_size} samples")
        end = time.time()
        logger.log("sampling complete")

        samples_arr = np.concatenate(all_samples, axis=0)[: self.num_samples]
        states_arr = np.concatenate(all_states, axis=0)[: self.num_samples]
        actions_arr = np.concatenate(all_actions, axis=0)[: self.num_samples]
        next_states_arr = np.concatenate(all_next_states, axis=0)[: self.num_samples]

        mse = ((next_states_arr - samples_arr) ** 2).mean(axis=0)
        logger.log(f"MSE between true next_states and sampled next_states: {mse}")
        logger.log(f"{self.num_samples} sampled in {end - start:.4f} sec")

        if save:
            if dist.get_rank() == 0:
                logger.log("Saving samples...")
                out_path = Path(self.sample_dir, f"{self.num_samples}_samples.npz")
                logger.log(f"saving to {out_path}")
                np.savez(
                    out_path,
                    states=states_arr,
                    actions=actions_arr,
                    true_next_states=next_states_arr,
                    sampled_next_states=samples_arr,
                )
                logger.log(f"Samples saved {self.sample_dir}.")

            dist.barrier()

        return samples_arr

    def _sample_image(self, sampling_fn, save):
        all_samples = []
        start = time.time()
        while len(all_samples) * self.batch_size < self.num_samples:
            if isinstance(self, SDESampler):
                sample, n = sampling_fn()(self.model)
            else:
                sample = sampling_fn()
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

            gathered_samples = [
                torch.zeros_like(sample) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

            logger.log(f"created {len(all_samples) * self.batch_size} samples")
        end = time.time()
        logger.log(f"{self.num_samples} sampled in {end - start:.4f} sec")
        samples_arr = np.concatenate(all_samples, axis=0)[: self.num_samples]
        logger.log("Sampling complete")
        if save:
            if dist.get_rank() == 0:
                logger.log("Saving samples...")
                shape_str = "x".join([str(x) for x in samples_arr.shape])
                out_path = os.path.join(self.sample_dir, f"samples_{shape_str}.npz")
                logger.log(f"saving to {out_path}")
                np.savez(out_path, samples_arr)
                logger.log(f"Samples saved {self.sample_dir}.")
            dist.barrier()
        return samples_arr

    def sample(self, save=True):
        if hasattr(self.dataset_cfg, "image_size"):
            sampling_fn = self._get_sample_fn_image()
            logger.log("sampling...")
            self._sample_image(sampling_fn, save)
        else:
            sampling_fn = self._get_sample_fn_dynamics()
            logger.log("sampling...")
            self._sample_dynamics(sampling_fn, save)


class DDPMSampler(Sampler):
    def __init__(self, use_ddim, clip_denoised, **kwargs):
        super().__init__(**kwargs)
        self.use_ddim = use_ddim
        self.clip_denoised = clip_denoised

    def _get_sample_fn_dynamics(self):
        sample_fn = (
            self.diffusion.p_sample_loop
            if not self.use_ddim
            else self.diffusion.ddim_sample_loop
        )

        sample_fn_wrapper = functools.partial(
            sample_fn,
            self.model,
            (self.batch_size, self.state_dim),
            clip_denoised=self.clip_denoised,
        )

        return sample_fn_wrapper

    def _get_sample_fn_image(self):
        model_kwargs = {}
        sample_fn = (
            self.diffusion.p_sample_loop
            if not self.use_ddim
            else self.diffusion.ddim_sample_loop
        )
        if self.model_cfg.class_cond:
            classes = torch.randint(
                low=0,
                high=NUM_CLASSES,
                size=(self.batch_size,),
                device=dist_util.DistUtil.dev(),
            )
            model_kwargs["y"] = classes
        sample_fn_wrapper = functools.partial(
            sample_fn,
            self.model,
            (
                self.batch_size,
                3,
                self.dataset_cfg.image_size,
                self.dataset_cfg.image_size,
            ),
            clip_denoised=self.clip_denoised,
            model_kwargs=model_kwargs,
        )
        return sample_fn_wrapper


class CMSampler(Sampler):
    def __init__(
        self,
        generator,
        sampler,
        s_churn,
        s_tmin,
        s_tmax,
        s_noise,
        steps,
        seed,
        ts,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.sampler = sampler
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.steps = steps
        self.seed = seed
        self.ts = ts

    def _get_sample_fn_image(self):
        model_kwargs = {}
        if "consistency" in self.diffusion.training_mode:
            distillation = True
        else:
            distillation = False
        if self.sampler == "multistep":
            assert len(self.ts) > 0
            ts = tuple(int(x) for x in self.ts.split(","))
        else:
            ts = None
        generator = get_generator(
            self.cm_sampler.generator, self.num_samples, self.cm_sampler.seed
        )
        if self.model.class_cond:
            classes = torch.randint(
                low=0,
                high=NUM_CLASSES,
                size=(self.batch_size,),
                device=dist_util.DistUtil.dev(),
            )
            model_kwargs["y"] = classes
        sample_fn_wrapper = functools.partial(
            karras_sample,
            self.diffusion,
            self.model,
            (
                self.batch_size,
                3,
                self.dataset.image_size,
                self.dataset.image_size,
            ),
            steps=self.cm_sampler.steps,
            device=dist_util.DistUtil.dev(),
            clip_denoised=self.cm_sampler.clip_denoised,
            sampler=self.cm_sampler.sampler,
            sigma_min=self.diffusion.sigma_min,
            sigma_max=self.diffusion.sigma_max,
            s_churn=self.s_churn,
            s_tmin=self.s_tmin,
            s_tmax=self.s_tmax,
            s_noise=self.s_noise,
            generator=generator,
            ts=ts,
            model_kwargs=model_kwargs,
        )
        return sample_fn_wrapper

    def _get_sample_fn_dynamics(self):
        model_kwargs = {}
        if self.sampler == "multistep":
            assert len(self.ts) > 0
            ts = tuple(int(x) for x in self.ts.split(","))
        else:
            ts = None
        generator = get_generator(
            self.cm_sampler.generator, self.num_samples, self.cm_sampler.seed
        )
        if self.model.class_cond:
            classes = torch.randint(
                low=0,
                high=NUM_CLASSES,
                size=(self.batch_size,),
                device=dist_util.DistUtil.dev(),
            )
            model_kwargs["y"] = classes
        sample_fn_wrapper = functools.partial(
            karras_sample,
            self.diffusion,
            self.model,
            (self.batch_size, self.state_dim),
            steps=self.cm_sampler.steps,
            device=dist_util.DistUtil.dev(),
            clip_denoised=self.cm_sampler.clip_denoised,
            sampler=self.cm_sampler.sampler,
            sigma_min=self.diffusion.sigma_min,
            sigma_max=self.diffusion.sigma_max,
            s_churn=self.s_churn,
            s_tmin=self.s_tmin,
            s_tmax=self.s_tmax,
            s_noise=self.s_noise,
            generator=generator,
            ts=ts,
            model_kwargs=model_kwargs,
        )
        return sample_fn_wrapper


class SDESampler(Sampler):
    def __init__(self, sde_sampler, **kwargs):
        super().__init__(**kwargs)
        self.sde_sampler = sde_sampler

    def _get_sample_fn_image(self):
        sampling_eps = 1e-3
        if isinstance(self.diffusion, VESDE):
            sampling_eps = 1e-5
        sampling_shape = (
            self.batch_size,
            3,
            self.dataset.image_size,
            self.dataset.image_size,
        )
        inverse_scaler = lambda x: (x + 1.0) / 2.0 if self.model.data_centered else x
        sampling_fn = functools.partial(
            sde_sampling.get_sampling_fn,
            self.sde_sampler,
            self.diffusion,
            sampling_shape,
            inverse_scaler,
            sampling_eps,
            continuous=self.diffusion.continuous,
            device=dist_util.DistUtil.dev(),
        )
        return sampling_fn

    def _get_sample_fn_dynamics(self):
        sampling_eps = 1e-3
        if isinstance(self.diffusion, VESDE):
            sampling_eps = 1e-5
        sampling_shape = (self.batch_size, self.state_dim)
        inverse_scaler = lambda x: x
        sampling_fn = functools.partial(
            sde_sampling.get_sampling_fn,
            self.sde_sampler,
            self.diffusion,
            sampling_shape,
            inverse_scaler,
            sampling_eps,
            continuous=self.diffusion.continuous,
            device=dist_util.DistUtil.dev(),
        )
        return sampling_fn
