import numpy as np
import torch


def load_tokens(filename):
    np_tokens = np.load(filename)
    torch_tokens = torch.tensor(np_tokens, dtype=torch.long)
    return torch_tokens
