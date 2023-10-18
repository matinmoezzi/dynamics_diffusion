# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop, RAdam
from dynamics_diffusion import dist_util, sde_sampling
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from dynamics_diffusion.gaussian_diffusion import ModelMeanType

from dynamics_diffusion.mlp import MLP
from dynamics_diffusion.resample import (
    LossAwareSampler,
    UniformSampler,
    create_named_schedule_sampler,
)
from dynamics_diffusion.sde import VESDE, VPSDE, subVPSDE
from dynamics_diffusion.sde_models.ema import ExponentialMovingAverage
from dynamics_diffusion.train_util import _expand_tensor_shape
from sacsvg.logger import Logger


def get_opt(optimizer_name, model_params, lr=0.001):
    """
    Instantiate a torch optimizer based on the optimizer name and model parameters.

    Args:
    optimizer_name (str): Name of the optimizer to use. Should be one of 'Adam', 'AdamW', 'RAdam', 'SGD', 'RMSprop'.
    model_params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
    lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
    torch.optim.Optimizer: An instantiated optimizer object.

    Raises:
    ValueError: If the optimizer_name is not recognized.
    """
    optimizers = {
        "Adam": Adam(model_params, lr=lr),
        "AdamW": AdamW(model_params, lr=lr),
        "RAdam": RAdam(model_params, lr=lr),
        "SGD": SGD(model_params, lr=lr),
        "RMSprop": RMSprop(model_params, lr=lr),
    }

    if optimizer_name in optimizers:
        return optimizers[optimizer_name]
    else:
        raise ValueError(
            f"Optimizer '{optimizer_name}' not recognized. Choose from 'Adam', 'AdamW', 'RAdam', 'SGD', 'RMSprop'."
        )


class DiffusionDx(nn.Module):
    def __init__(
        self,
        model,
        diffusion,
        env_name,
        obs_dim,
        action_dim,
        horizon,
        device,
        detach_xt,
        clip_grad_norm,
        lr,
        opt_name,
        use_fp16,
        ema_rate,
        fp16_scale_growth,
    ):
        super().__init__()

        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = device
        self.detach_xt = detach_xt
        self.clip_grad_norm = clip_grad_norm
        self.opt_name = opt_name
        self.diffusion = diffusion
        self.use_fp16 = use_fp16
        self.logger = Logger.get_logger()

        if isinstance(model, torch.nn.Module):
            self.model = model
        elif isinstance(model, functools.partial):
            if model.func == MLP:
                self.model = model(
                    x_dim=self.obs_dim, cond_dim=self.obs_dim + self.action_dim
                )
        else:
            raise ValueError(f"Unknown model {model}")
        self.model = self.model.to(dist_util.DistUtil.dev())

        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.ema = [
            ExponentialMovingAverage(self.model.parameters(), decay=rate)
            for rate in range(len(self.ema_rate))
        ]

        # Manually freeze the goal locations
        if env_name == "gym_petsReacher":
            self.freeze_dims = torch.LongTensor([7, 8, 9])
        elif env_name == "gym_petsPusher":
            self.freeze_dims = torch.LongTensor([20, 21, 22])
        else:
            self.freeze_dims = None

        self.opt = get_opt(self.opt_name, list(self.model.parameters()), lr=lr)

        if dist_util.DistUtil.device == "cpu":
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            self.model = self.ddp_model

        elif torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.DistUtil.dev()],
                output_device=dist_util.DistUtil.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            self.model = self.ddp_model
        else:
            if dist.get_world_size() > 1:
                self.logger.log(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def __getstate__(self):
        snapshot = {}
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        snapshot["model_state"] = raw_model.state_dict()
        snapshot["opt_state"] = self.opt.state_dict()
        for rate, ema in zip(self.ema_rate, self.ema):
            snapshot[f"ema_{rate}"] = ema.state_dict()
        return snapshot

    def __setstate__(self, d):
        self.model.load_state_dict(d["model_state"])
        self.opt.load_state_dict(d["opt_state"])
        for rate, ema in zip(self.ema_rate, self.ema):
            ema.load_state_dict(d[f"ema_{rate}"])

    def forward(self, x, us):
        return self.unroll(x, us)


class DDPMDx(DiffusionDx):
    def __init__(self, use_ddim, clip_denoised, schedule_sampler, **kwargs):
        super().__init__(**kwargs)
        self.use_ddim = use_ddim
        self.clip_denoised = clip_denoised
        schedule_sampler = schedule_sampler or UniformSampler(self.diffusion)
        self.schedule_sampler = create_named_schedule_sampler(
            schedule_sampler, self.diffusion
        )

    def unroll_policy(self, init_x, policy, sample=True, last_u=True, detach_xt=False):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.freeze_dims is not None:
            obs_frozen = init_x[:, self.freeze_dims]

        if self.rec_num_layers > 0:
            h = self.init_hidden_state(init_x)

        pred_xs = []
        us = []
        log_p_us = []
        xt = init_x
        for t in range(self.horizon - 1):
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

            if detach_xt:
                xt = xt.detach()

            diffusion = self.diffusion
            sample_fn = (
                diffusion.p_sample_loop
                if not self.use_ddim
                else diffusion.ddim_sample_loop
            )

            sample_fn_wrapper = functools.partial(
                sample_fn,
                self.ddp_model.module,
                (n_batch, self.obs_dim),
                clip_denoised=self.clip_denoised,
            )
            dx_sample = sample_fn_wrapper(model_kwargs={"state": xt, "action": ut})

            # Assume that s_{t+1} = s_t + dx_t
            xtp1 = xt + dx_sample

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen

            pred_xs.append(xtp1)
            xt = xtp1

        if last_u:
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

        us = torch.stack(us)
        log_p_us = torch.stack(log_p_us).squeeze(2)
        if self.horizon <= 1:
            pred_xs = torch.empty(0, n_batch, self.obs_dim).to(init_x.device)
        else:
            pred_xs = torch.stack(pred_xs)

        return us, log_p_us, pred_xs

    def unroll(self, x, us, detach_xt=False):
        assert x.dim() == 2
        assert us.dim() == 3
        n_batch = x.size(0)
        assert us.size(1) == n_batch

        if self.freeze_dims is not None:
            obs_frozen = x[:, self.freeze_dims]

        pred_xs = []
        xt = x
        for t in range(us.size(0)):
            ut = us[t]

            if detach_xt:
                xt = xt.detach()

            diffusion = self.diffusion
            sample_fn = (
                diffusion.p_sample_loop
                if not self.use_ddim
                else diffusion.ddim_sample_loop
            )

            dx_sample = sample_fn(
                self.ddp_model.module,
                (n_batch, self.obs_dim),
                clip_denoised=self.clip_denoised,
                model_kwargs={"state": xt, "action": ut},
            )

            # Assume that s_{t+1} = s_t + dx_t
            xtp1 = xt + dx_sample

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen

            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = torch.stack(pred_xs)

        return pred_xs

    def update_step(self, obs, action, reward, logger, step):
        assert obs.dim() == 3
        T, batch_size, _ = obs.shape

        self.opt.zero_grad()
        obs_losses = []
        diffusion_losses = []
        for horizon in range(T - 1):
            t, weights = self.schedule_sampler.sample(
                batch_size, dist_util.DistUtil.dev()
            )

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                obs[horizon + 1],
                t,
                model_kwargs={"state": obs[horizon], "action": action[horizon]},
            )
            losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            loss = (losses["loss"] * weights).mean()

            diffusion_losses.append(loss)

            if self.diffusion.model_mean_type != ModelMeanType.START_X:
                if self.freeze_dims is not None:
                    obs_frozen = obs[horizon][:, self.freeze_dims]
                sample_fn = (
                    self.diffusion.p_sample_loop
                    if not self.use_ddim
                    else self.diffusion.ddim_sample_loop
                )

                dx_sample = sample_fn(
                    self.ddp_model.module,
                    (batch_size, self.obs_dim),
                    clip_denoised=self.clip_denoised,
                    model_kwargs={"state": obs[horizon], "action": action[horizon]},
                )

                xtp1 = obs[horizon] + dx_sample

                if self.freeze_dims is not None:
                    xtp1[:, self.freeze_dims] = obs_frozen

                target_obs = obs[horizon + 1]

                assert xtp1.size() == target_obs.size()

                obs_loss = F.mse_loss(xtp1, target_obs, reduction="mean")

            else:
                obs_loss = losses["mse"].mean()

            obs_losses.append(obs_loss)

            loss.backward()

        if self.clip_grad_norm is not None:
            assert len(self.opt.param_groups) == 1
            params = self.opt.param_groups[0]["params"]
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)

        self.opt.step()

        for ema in self.ema:
            ema.update(self.model.parameters())

        step_obs_loss = torch.stack(obs_losses).mean().item()
        step_diffusion_loss = torch.stack(diffusion_losses).mean().item()

        logger.log("train_model/obs_loss", step_obs_loss, step)
        logger.log("train_diffusion/loss", step_diffusion_loss, step)

        return (
            torch.stack(diffusion_losses).mean().item(),
            torch.stack(obs_losses).mean().item(),
        )


class ScoreSDEDx(DiffusionDx):
    def __init__(
        self, sde_sampler, sde_continuous, sde_likelihood_weighting, sde_eps, **kwargs
    ):
        super().__init__(**kwargs)

        self.sde_sampler = sde_sampler
        self.sde_continuous = sde_continuous
        self.sde_eps = sde_eps
        self.sde_likelihood_weighting = sde_likelihood_weighting

    def unroll_policy(self, init_x, policy, sample=True, last_u=True, detach_xt=False):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.freeze_dims is not None:
            obs_frozen = init_x[:, self.freeze_dims]

        pred_xs = []
        us = []
        log_p_us = []
        xt = init_x
        for t in range(self.horizon - 1):
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

            if detach_xt:
                xt = xt.detach()

            # State-action embedding
            # xu_emb = self.xu_enc(xut).unsqueeze(0)

            # Sampling ScoreSDEDx
            sde = self.diffusion
            sampling_eps = 1e-3
            if isinstance(sde, VESDE):
                sampling_eps = 1e-5
            sampling_shape = (n_batch, self.obs_dim)
            inverse_scaler = lambda x: x
            sampling_fn = functools.partial(
                sde_sampling.get_sampling_fn,
                self.sde_sampler,
                sde,
                sampling_shape,
                inverse_scaler,
                sampling_eps,
                continuous=self.sde_continuous,
                device=dist_util.DistUtil.dev(),
            )
            model_kwargs = {"state": xt, "action": ut}
            dx_sample, n = sampling_fn(model_kwargs=model_kwargs)(self.model)

            # Assume that s_{t+1} = s_t + dx_t
            xtp1 = xt + dx_sample
            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen

            pred_xs.append(xtp1)
            xt = xtp1

        if last_u:
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

        us = torch.stack(us)
        log_p_us = torch.stack(log_p_us).squeeze(2)
        if self.horizon <= 1:
            pred_xs = torch.empty(0, n_batch, self.obs_dim).to(init_x.device)
        else:
            pred_xs = torch.stack(pred_xs)

        return us, log_p_us, pred_xs

    def unroll(self, x, us, detach_xt=False):
        assert x.dim() == 2
        assert us.dim() == 3
        n_batch = x.size(0)
        assert us.size(1) == n_batch

        if self.freeze_dims is not None:
            obs_frozen = x[:, self.freeze_dims]

        pred_xs = []
        xt = x
        for t in range(us.size(0)):
            ut = us[t]

            if detach_xt:
                xt = xt.detach()

            # State-action embedding
            # xu_emb = self.xu_enc(xut).unsqueeze(0)

            # Sampling ScoreSDEDx
            sde = self.diffusion
            sampling_eps = 1e-3
            if isinstance(sde, VESDE):
                sampling_eps = 1e-5
            sampling_shape = (n_batch, self.obs_dim)
            inverse_scaler = lambda x: x
            sampling_fn = functools.partial(
                sde_sampling.get_sampling_fn,
                self.sde_sampler,
                sde,
                sampling_shape,
                inverse_scaler,
                sampling_eps,
                continuous=self.sde_continuous,
                device=dist_util.DistUtil.dev(),
            )
            model_kwargs = {"state": xt, "action": ut}
            dx_sample, n = sampling_fn(model_kwargs=model_kwargs)(self.model)

            # Assume that s_{t+1} = s_t + dx_t
            xtp1 = xt + dx_sample

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen
            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = torch.stack(pred_xs)

        return pred_xs

    def update_step(self, obs, action, reward, logger, step):
        assert obs.dim() == 3
        T, batch_size, _ = obs.shape

        pred_obs = self.unroll(obs[0], action[:-1], detach_xt=self.detach_xt)
        target_obs = obs[1:]
        assert pred_obs.size() == target_obs.size()

        ttt = F.mse_loss(pred_obs, target_obs)
        obs_loss = F.mse_loss(pred_obs, target_obs, reduction="mean")

        self.opt.zero_grad()
        obs_losses = []
        diffusion_losses = []
        for horizon in range(T - 1):
            model_kwargs = {"state": obs[horizon], "action": action[horizon]}

            # Observation Loss
            if self.freeze_dims is not None:
                obs_frozen = obs[horizon][:, self.freeze_dims]
            sampling_eps = 1e-3
            sde = self.diffusion
            if isinstance(sde, VESDE):
                sampling_eps = 1e-5
            sampling_shape = (batch_size, self.obs_dim)
            inverse_scaler = lambda x: x
            sampling_fn = functools.partial(
                sde_sampling.get_sampling_fn,
                self.sde_sampler,
                sde,
                sampling_shape,
                inverse_scaler,
                sampling_eps,
                continuous=self.sde_continuous,
                device=dist_util.DistUtil.dev(),
            )
            dx_sample, n = sampling_fn(model_kwargs=model_kwargs)(self.model)

            # Assume that s_{t+1} = s_t + dx_t
            xtp1 = obs[horizon] + dx_sample

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen

            target_obs = obs[horizon + 1]

            obs_loss = F.mse_loss(xtp1, target_obs, reduction="mean")

            obs_losses.append(obs_loss)

            # Score SDE Score-matching Loss
            if self.sde_continuous:
                t = (
                    torch.rand(batch_size, device=dist_util.DistUtil.dev())
                    * (self.diffusion.T - self.sde_eps)
                    + self.sde_eps
                )
                compute_losses = functools.partial(
                    self._continuous_loss, obs[horizon + 1], t, model_kwargs
                )
            else:
                assert (
                    not self.sde_likelihood_weighting
                ), "Likelihood weighting is not supported for original SMLD/DDPM training."
                t = torch.randint(
                    0,
                    self.diffusion.N,
                    (batch_size,),
                    device=dist_util.DistUtil.dev(),
                )
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    model=self.ddp_model,
                    batch=obs[horizon + 1],
                    labels=t,
                    model_kwargs=model_kwargs,
                )

            losses = compute_losses()
            loss = (losses["loss"]).mean()

            diffusion_losses.append(loss)

            loss.backward()

        if self.clip_grad_norm is not None:
            assert len(self.opt.param_groups) == 1
            params = self.opt.param_groups[0]["params"]
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.opt.step()
        for ema in self.ema:
            ema.update(self.model.parameters())

        step_obs_loss = torch.stack(obs_losses).mean().item()
        step_diffusion_loss = torch.stack(diffusion_losses).mean().item()

        logger.log("train_model/obs_loss", step_obs_loss, step)
        logger.log("train_diffusion/loss", step_diffusion_loss, step)

        return (
            torch.stack(diffusion_losses).mean().item(),
            torch.stack(obs_losses).mean().item(),
        )

    def _continuous_loss(self, batch, t, model_kwargs):
        reduce_op = (
            torch.mean
            if self.diffusion.reduce_mean
            else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        )

        z = torch.randn_like(batch)

        mean, std = self.diffusion.marginal_prob(batch, t)
        perturbed_data = mean + _expand_tensor_shape(std, z.shape) * z
        score = self._get_score(perturbed_data, t, model_kwargs)

        if not self.sde_likelihood_weighting:
            losses = torch.square(score * _expand_tensor_shape(std, z.shape) + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = self.diffusion.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / _expand_tensor_shape(std, score.shape))
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        return {"loss": losses}

    def _get_score(self, x, t, model_kwargs=None):
        if isinstance(self.diffusion, VPSDE) or isinstance(self.diffusion, subVPSDE):
            if self.continuous or isinstance(self.diffusion, subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = self.ddp_model(x, labels, **model_kwargs)
                std = self.diffusion.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (self.diffusion.N - 1)
                score = self.ddp_model(x, labels, **model_kwargs)
                std = self.diffusion.sqrt_1m_alphas_cumprod.to(labels.device)[
                    labels.long()
                ]

            score = -score / _expand_tensor_shape(std, score.shape)
            return score

        elif isinstance(self.diffusion, VESDE):
            if self.sde_continuous:
                labels = self.diffusion.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = self.diffusion.T - t
                labels *= self.diffusion.N - 1
                labels = torch.round(labels).long()

            score = self.ddp_model(x, labels, **model_kwargs)
            return score

        else:
            raise NotImplementedError(
                f"SDE class {self.diffusion.__class__.__name__} not yet supported."
            )
