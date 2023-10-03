import copy
import hydra
import numpy as np
import torch as th
from dynamics_diffusion import sde_sampling
from dynamics_diffusion.resample import UniformSampler, create_named_schedule_sampler

from dynamics_diffusion.sde import SDE, VESDE, VPSDE, subVPSDE


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def _expand_tensor_shape(x, shape):
    while len(x.shape) < len(shape):
        x = x[..., None]
    return x


def log_loss_dict(logger, step, diffusion, ts, losses):
    for key, values in losses.items():
        logger.log("train/diffusion_" + key, values.mean().item(), step)
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.log(f"train/diffusion_{key}_q{quartile}", sub_loss, step)


class DiffusionDx(th.nn.Module):
    def __init__(
        self,
        env_name,
        obs_dim,
        action_dim,
        horizon,
        device,
        detach_xt,
        clip_grad_norm,
        lr,
        model,
        diffusion,
        opt_name,
        ema_rate,
        schedule_sampler,
        weight_decay,
        learn_sigma,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        super().__init__()
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = device
        self.detach_xt = detach_xt
        self.clip_grad_norm = clip_grad_norm
        self.lr = lr
        self.learn_sigma = learn_sigma
        self.freeze_dims = None

        self.model = hydra.utils.instantiate(
            model.target,
            x_dim=self.obs_dim,
            cond_dim=self.obs_dim + self.action_dim,
            learn_sigma=self.learn_sigma,
        )
        self.model = self.model.to(self.device)
        self.diffusion = hydra.utils.call(diffusion.target)
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        if not isinstance(self.diffusion, SDE):
            self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        else:
            self.schedule_sampler = schedule_sampler

        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler, self.diffusion
        )
        self.weight_decay = weight_decay

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.opt_name = opt_name.lower()
        self.opt = self._get_optimizer()

        self.ema_params = [
            copy.deepcopy(list(self.model.parameters()))
            for _ in range(len(self.ema_rate))
        ]

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
            model_kwargs = {"state": xt, "action": ut}
            xtp1 = self.diffusion.p_sample_loop(
                self.model,
                (n_batch, self.obs_dim),
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
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

        us = th.stack(us)
        log_p_us = th.stack(log_p_us).squeeze(2)
        if self.horizon <= 1:
            pred_xs = th.empty(0, n_batch, self.obs_dim).to(init_x.device)
        else:
            pred_xs = th.stack(pred_xs)

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

            model_kwargs = {"state": xt, "action": ut}
            xtp1 = self.diffusion.p_sample_loop(
                self.model,
                (n_batch, self.obs_dim),
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
            )

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen
            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = th.stack(pred_xs)

        return pred_xs

    def update_step(self, obs, action, reward, logger, step):
        assert obs.dim() == 3
        T, batch_size, _ = obs.shape

        t, weights = self.schedule_sampler.sample(batch_size, self.device)

        acc_loss = []
        for i in range(T - 1):
            losses = self.diffusion.training_losses(
                self.model, obs[i + 1], t, {"state": obs[i], "action": action[i]}
            )
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                logger,
                step,
                self.diffusion,
                t,
                {k: v * weights for k, v in losses.items()},
            )
            acc_loss.append(loss)
        obs_loss = th.stack(acc_loss).sum(dim=-1)

        logger.log("train/model_obs_loss", obs_loss.item(), step)

        self.opt.zero_grad()
        obs_loss.backward()
        if self.clip_grad_norm is not None:
            assert len(self.opt.param_groups) == 1
            params = self.opt.param_groups[0]["params"]
            th.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.opt.step()

        self._update_ema()

        return obs_loss.item()

    def forward(self, x, us):
        return self.unroll(x, us)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, list(self.model.parameters()), rate=rate)

    def _get_optimizer(self):
        if self.opt_name == "adam":
            return th.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        elif self.opt_name == "adamax":
            return th.optim.Adamax(
                self.model.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        elif self.opt_name == "adamw":
            return th.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        elif self.opt_name == "rmsprop":
            return th.optim.RMSprop(
                self.model.parameters(),
                lr=self.lr,
                alpha=self.alpha,
                eps=self.eps,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                centered=self.centered,
            )
        elif self.opt_name == "sgd":
            return th.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                dampening=self.dampening,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.opt_name}")


class ScoreSDEDx(DiffusionDx):
    def __init__(
        self,
        env_name,
        obs_dim,
        action_dim,
        horizon,
        device,
        detach_xt,
        clip_grad_norm,
        lr,
        model,
        diffusion,
        opt_name,
        ema_rate,
        schedule_sampler,
        weight_decay,
        continuous,
        likelihood_weighting,
        sde_sampler,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        super().__init__(
            env_name,
            obs_dim,
            action_dim,
            horizon,
            device,
            detach_xt,
            clip_grad_norm,
            lr,
            model,
            diffusion,
            opt_name,
            ema_rate,
            schedule_sampler,
            weight_decay,
            False,
            beta1,
            beta2,
            eps,
        )
        self.continuous = continuous
        self.likelihood_weighting = likelihood_weighting
        self.sde_sampler = sde_sampler

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
            model_kwargs = {"state": xt, "action": ut}

            sampling_eps = 1e-3
            if isinstance(self.diffusion, VESDE):
                sampling_eps = 1e-5
            sampling_shape = (n_batch, self.obs_dim)
            inverse_scaler = lambda x: x
            xtp1, n = sde_sampling.get_sampling_fn(
                self.sde_sampler,
                self.diffusion,
                sampling_shape,
                inverse_scaler,
                sampling_eps,
                self.continuous,
                device=self.device,
                model_kwargs=model_kwargs,
            )(self.model)

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

        us = th.stack(us)
        log_p_us = th.stack(log_p_us).squeeze(2)
        if self.horizon <= 1:
            pred_xs = th.empty(0, n_batch, self.obs_dim).to(init_x.device)
        else:
            pred_xs = th.stack(pred_xs)

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

            model_kwargs = {"state": xt, "action": ut}
            sampling_eps = 1e-3
            if isinstance(self.diffusion, VESDE):
                sampling_eps = 1e-5
            sampling_shape = (n_batch, self.obs_dim)
            inverse_scaler = lambda x: x
            xtp1, n = sde_sampling.get_sampling_fn(
                self.sde_sampler,
                self.diffusion,
                sampling_shape,
                inverse_scaler,
                sampling_eps,
                self.continuous,
                model_kwargs=model_kwargs,
            )(self.model)

            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen
            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = th.stack(pred_xs)

        return pred_xs

    def update_step(self, obs, action, reward, logger, step):
        assert obs.dim() == 3
        T, batch_size, _ = obs.shape

        acc_loss = []
        for i in range(T - 1):
            if self.continuous:
                t = (
                    th.rand(batch_size, device=self.device)
                    * (self.diffusion.T - self.eps)
                    + self.eps
                )
                losses = self._continuous_loss(
                    obs[i + 1], t, {"state": obs[i], "action": action[i]}
                )
            else:
                assert (
                    not self.likelihood_weighting
                ), "Likelihood weighting is not supported for original SMLD/DDPM training."
                t = th.randint(0, self.diffusion.N, (batch_size,), device=self.device)
                losses = self.diffusion.training_losses(
                    model=self.model,
                    batch=obs[i + 1],
                    labels=t,
                    model_kwargs={"state": obs[i], "action": action[i]},
                )
            loss = (losses["loss"]).mean()
            log_loss_dict(
                logger,
                step,
                self.diffusion,
                t,
                {k: v for k, v in losses.items()},
            )
            acc_loss.append(loss)

        obs_loss = th.stack(acc_loss).sum(dim=-1)

        logger.log("train/model_obs_loss", obs_loss.item(), step)

        self.opt.zero_grad()
        obs_loss.backward()
        if self.clip_grad_norm is not None:
            assert len(self.opt.param_groups) == 1
            params = self.opt.param_groups[0]["params"]
            th.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.opt.step()

        self._update_ema()

        return obs_loss.item()

    def _continuous_loss(self, batch, t, model_kwargs):
        reduce_op = (
            th.mean
            if self.diffusion.reduce_mean
            else lambda *args, **kwargs: 0.5 * th.sum(*args, **kwargs)
        )

        z = th.randn_like(batch)

        mean, std = self.diffusion.marginal_prob(batch, t)
        perturbed_data = mean + _expand_tensor_shape(std, z.shape) * z
        score = self._get_score(perturbed_data, t, model_kwargs)

        if not self.likelihood_weighting:
            losses = th.square(score * _expand_tensor_shape(std, z.shape) + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = self.diffusion.sde(th.zeros_like(batch), t)[1] ** 2
            losses = th.square(score + z / _expand_tensor_shape(std, score.shape))
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
                std = self.diffusion.marginal_prob(th.zeros_like(x), t)[1]
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
            if self.continuous:
                labels = self.diffusion.marginal_prob(th.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = self.diffusion.T - t
                labels *= self.diffusion.N - 1
                labels = th.round(labels).long()

            score = self.ddp_model(x, labels, **model_kwargs)
            return score

        else:
            raise NotImplementedError(
                f"SDE class {self.diffusion.__class__.__name__} not yet supported."
            )
