import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor):
        """forward throught the attention block x: (B, T, C)"""  # what is C ? : C is
        B, T, C = x.shape
        qkv = self.c_attn(x)  # (B, n_head, T, 3 * n_embd)
        q, k, v = torch.split(
            qkv, split_size_or_sections=self.n_embd, dim=-1
        )  # each of size (B, T, n_embd)

        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(
            1, 2
        )  # (B, n_head, T, hs)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

        # att = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, n_head, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(wei, dim=-1)
        # out = wei @ v  # (B, n_head, T, T) @ (B, n_head, T, hs) -> (B, n_head, T, hs)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )  # -> (B, n_head, T, hs)

        out = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # (B, T, n_head, hs) -> (B, T, C)
        out = self.c_proj(out)

        return out
