from torch import nn
from torch.nn.functional import silu


class SwigLU(nn.Module):
    def forward(self, x):
        gate, value = x.chunk(2, dim=-1)
        out = silu(gate) * value
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.n_embd
        if config.activation == "gelu":
            hidden_width = 4 * width
            self.c_fc = nn.Linear(width, hidden_width)
            self.activation = nn.GELU(approximate="tanh")
        else:
            hidden_width = (8 * width) // 3
            self.c_fc = nn.Linear(width, hidden_width * 2)
            self.activation = SwigLU()
        self.c_proj = nn.Linear(hidden_width, width)
        self.c_proj.RESIDUAL_SCALE = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x
