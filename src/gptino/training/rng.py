import random

import numpy as np
import torch


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


def get_rng_state(device):
    device = torch.device(device)
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state, device):
    device = torch.device(device)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoint map_location can move RNG tensors to the training device,
    # but the RNG setters require CPU byte tensors.
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["mps"].cpu())
