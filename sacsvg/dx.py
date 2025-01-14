# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.distributed as dist

from dynamics_diffusion import dist_util, logger

from . import utils


class SeqNN(nn.Module):
    def __init__(
        self,
        env_name,
        obs_dim,
        action_dim,
        horizon,
        detach_xt,
        clip_grad_norm,
        xu_enc_hidden_dim,
        xu_enc_hidden_depth,
        x_dec_hidden_dim,
        x_dec_hidden_depth,
        rec_type,
        rec_latent_dim,
        rec_num_layers
    ):
        super().__init__()

        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = dist_util.DistUtil.dev()
        self.detach_xt = detach_xt
        self.clip_grad_norm = clip_grad_norm

        # Manually freeze the goal locations
        if env_name == "gym_petsReacher":
            self.freeze_dims = torch.LongTensor([7, 8, 9])
        elif env_name == "gym_petsPusher":
            self.freeze_dims = torch.LongTensor([20, 21, 22])
        else:
            self.freeze_dims = None

        self.rec_type = rec_type
        self.rec_num_layers = rec_num_layers
        self.rec_latent_dim = rec_latent_dim

        self.xu_enc = utils.mlp(
            obs_dim + action_dim, xu_enc_hidden_dim, rec_latent_dim, xu_enc_hidden_depth
        )
        self.x_dec = utils.mlp(
            rec_latent_dim, x_dec_hidden_dim, obs_dim, x_dec_hidden_depth
        )

        self.apply(utils.weight_init)  # Don't apply this to the recurrent unit.

        mods = [self.xu_enc, self.x_dec]

        if rec_num_layers > 0:
            if rec_type == "LSTM":
                self.rec = nn.LSTM(
                    rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers
                )
            elif rec_type == "GRU":
                self.rec = nn.GRU(
                    rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers
                )
            else:
                assert False
            mods.append(self.rec)

        self.params = list(utils.get_params(mods))

    def __getstate__(self):
        d = self.__dict__
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.rec.flatten_parameters()

    def init_hidden_state(self, init_x):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.rec_type == "LSTM":
            h = torch.zeros(
                self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device
            )
            c = torch.zeros_like(h)
            h = (h, c)
        elif self.rec_type == "GRU":
            h = torch.zeros(
                self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device
            )
        else:
            assert False

        return h

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

            xut = torch.cat((xt, ut), dim=1)
            xu_emb = self.xu_enc(xut).unsqueeze(0)
            if self.rec_num_layers > 0:
                xtp1_emb, h = self.rec(xu_emb, h)
            else:
                xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
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

        if self.rec_num_layers > 0:
            h = self.init_hidden_state(x)

        pred_xs = []
        xt = x
        for t in range(us.size(0)):
            ut = us[t]

            if detach_xt:
                xt = xt.detach()

            xut = torch.cat((xt, ut), dim=1)
            xu_emb = self.xu_enc(xut).unsqueeze(0)
            if self.rec_num_layers > 0:
                xtp1_emb, h = self.rec(xu_emb, h)
            else:
                xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen
            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = torch.stack(pred_xs)

        return pred_xs

    def forward(self, x, us, detach_xt=False):
        return self.unroll(x, us, detach_xt=detach_xt)


class SeqDx(nn.Module):
    def __init__(
        self,
        env_name,
        obs_dim,
        action_dim,
        horizon,
        detach_xt,
        clip_grad_norm,
        xu_enc_hidden_dim,
        xu_enc_hidden_depth,
        x_dec_hidden_dim,
        x_dec_hidden_depth,
        rec_type,
        rec_latent_dim,
        rec_num_layers,
        lr,
        abs_pos=False
    ):
        super().__init__()

        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = dist_util.DistUtil.dev()
        self.detach_xt = detach_xt
        self.clip_grad_norm = clip_grad_norm
        self.xu_enc_hidden_dim = xu_enc_hidden_dim
        self.xu_enc_hidden_depth = xu_enc_hidden_depth
        self.x_dec_hidden_dim = x_dec_hidden_dim
        self.x_dec_hidden_depth = x_dec_hidden_depth
        self.rec_type = rec_type
        self.rec_latent_dim = rec_latent_dim
        self.rec_num_layers = rec_num_layers
        self.lr = lr

        self.model = SeqNN(
            env_name,
            obs_dim,
            action_dim,
            horizon,
            detach_xt,
            clip_grad_norm,
            xu_enc_hidden_dim,
            xu_enc_hidden_depth,
            x_dec_hidden_dim,
            x_dec_hidden_depth,
            rec_type,
            rec_latent_dim,
            rec_num_layers,
        )
        self.model = self.model.to(dist_util.DistUtil.dev())

        # self.opt = torch.optim.Adam(self.model.params, lr=lr)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        if dist_util.DistUtil.device == "cpu":
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
            )
            self.model = self.ddp_model

        elif torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.DistUtil.dev()],
            )
            self.model = self.ddp_model
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def to(self, device):
        super().to(device)
        self.model.to(device)
        if self.use_ddp:
            self.ddp_model.to(device)
            self.ddp_model.device = torch.device(device)
            self.model.device = torch.device(device)
            self.ddp_model.device_ids = [torch.device(device).index]
            self.model.device_ids = [torch.device(device).index]
        return self

    def update_step(self, obs, action, reward, step):
        assert obs.dim() == 3
        T, batch_size, _ = obs.shape

        pred_obs = self.model(obs[0], action[:-1], detach_xt=self.detach_xt)
        target_obs = obs[1:]
        assert pred_obs.size() == target_obs.size()

        obs_loss = F.mse_loss(pred_obs, target_obs, reduction="mean")

        self.opt.zero_grad()
        obs_loss.backward()
        if self.clip_grad_norm is not None:
            assert len(self.opt.param_groups) == 1
            params = self.opt.param_groups[0]["params"]
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)

        self.opt.step()

        logger.logkv_mean("train_model/obs_loss", obs_loss, step)

        return obs_loss.item()

    def unroll_policy(self, init_x, policy, sample=True, last_u=True, detach_xt=False):
        if hasattr(self.model, "module"):
            return self.model.module.unroll_policy(
                init_x, policy, sample=sample, last_u=last_u, detach_xt=detach_xt
            )
        return self.model.unroll_policy(
            init_x, policy, sample=sample, last_u=last_u, detach_xt=detach_xt
        )

    def unroll(self, x, us, detach_xt=False):
        if hasattr(self.model, "module"):
            return self.model.module.unroll(x, us, detach_xt=detach_xt)
        return self.model.unroll(x, us, detach_xt=detach_xt)
