import torch
from torch import nn


class RoPE(nn.Module):
    def __init__(self,  dim, base = 10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0
        self.base = base
        self.cosine_mat = None
        self.sine_mat = None


    def _build_matrices(self, x):
        assert x.shape[-1] == self.dim
        if (self.cosine_mat is not None) and (x.device == self.cosine_mat.device) and (x.dtype == self.cosine_mat.dtype) and (x.shape[-2] == self.cosine_mat.shape[-2]):
            return


        pos_idx = torch.arange(0, x.shape[-2], dtype=torch.float32, device=x.device) # T
        theta = 1. / self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=x.device) / self.dim) # D/2 [theta_1, theta_2, ...,Theta_d/2]

        pos_theta = pos_idx[:, None] * theta[None, :] # T, D/2
        pos_theta = torch.cat([pos_theta, pos_theta], dim=-1) # T, D

        self.cosine_mat = pos_theta.cos().to(x.dtype)[None, None, :, :] #x, x, T, D
        self.sine_mat = pos_theta.sin().to(x.dtype)[None, None, :, :] #x, x, T, D

    def forward(self, x):
        self._build_matrices(x)
        d_2 = x.shape[-1] // 2
        x_ = torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)
        x_rope = x * self.cosine_mat + x_ * self.sine_mat
        return x_rope
