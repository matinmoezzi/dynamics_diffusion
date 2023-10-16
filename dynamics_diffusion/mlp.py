import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    """
    MLP Model
    """

    def __init__(self, x_dim, cond_dim, use_fp16, learn_sigma=False, t_dim=16):
        super(MLP, self).__init__()

        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.learn_sigma = learn_sigma

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = x_dim + cond_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
        )

        if self.learn_sigma:
            out_dim = 2 * x_dim
        else:
            out_dim = x_dim

        self.final_layer = nn.Linear(256, out_dim)

    def forward(self, x, time, state, action):
        state = state.type(self.dtype)
        action = action.type(self.dtype)
        time = time.type(self.dtype)
        x = x.type(self.dtype)
        t = self.time_mlp(time)
        x = torch.cat((x, t, state, action), dim=-1)
        x = self.mid_layer(x)

        return self.final_layer(x)

    def convert_to_fp16(self):
        self.half()

    def convert_to_fp32(self):
        self.float()
