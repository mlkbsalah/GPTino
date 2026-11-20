from dataclasses import dataclass
import os
import torch
from torch.distributed import init_process_group


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    return device


@dataclass
class DDPContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    master_process: bool
    device: str


def setup_ddp():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = get_device()
    return DDPContext(
        enabled=ddp,
        rank=ddp_rank,
        local_rank=ddp_local_rank,
        world_size=ddp_world_size,
        master_process=master_process,
        device=device,
    )
