from torch import nn

from .attention import CausalSelfAttention
from .mlp import MLP


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.normalization == "rmsnorm":
            self.ln_1 = nn.RMSNorm(config.n_embd)
            self.ln_2 = nn.RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x
