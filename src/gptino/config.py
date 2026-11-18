import os
from dataclasses import dataclass, replace

import torch
import yaml
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


@dataclass
class TrainConfig:
    data_dir: str
    model: dict
    seed: int
    train_tokens: int
    train_micro_batch_size: int
    train_batch_tokens: int
    val_tokens: int
    val_micro_batch_size: int
    context_length: int

    lr_scheduler: dict

    weight_decay: float
    grad_clip_norm: float

    validation_interval: int
    sample_interval: int
    save_interval: int
    save_checkpoints: bool

    sample: dict
    task_evaluation: dict | None = None


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_train_config(path, debug=False, world_size=1):
    raw = load_config(path)
    training = raw["training"]
    lr = training["lr_scheduler"]
    logging = training["logging"]
    optimizer = training["optimizer"]
    sample = training["sample"]

    cfg = TrainConfig(
        data_dir=raw["data"]["data_dir"],
        model=raw["model"],
        seed=training["seed"],
        train_micro_batch_size=training["train_micro_batch_size"],
        val_micro_batch_size=training["val_micro_batch_size"],
        context_length=raw["model"]["context_length"],
        train_batch_tokens=training["train_batch_tokens"],
        train_tokens=int(training["train_tokens"]),
        val_tokens=int(training["val_tokens"]),
        lr_scheduler=lr,
        weight_decay=optimizer["weight_decay"],
        grad_clip_norm=optimizer["grad_clip_norm"],
        validation_interval=logging["validation_interval"],
        sample_interval=logging["sample_interval"],
        save_interval=logging["save_interval"],
        save_checkpoints=True,
        sample=sample,
        task_evaluation=training.get("task_evaluation"),
    )

    return debug_config(cfg, world_size) if debug else cfg


def save_config(cfg: TrainConfig, path):
    raw = {
        "model": {**cfg.model, "context_length": cfg.context_length},
        "data": {"data_dir": cfg.data_dir},
        "training": {
            "seed": cfg.seed,
            "train_tokens": cfg.train_tokens,
            "train_batch_tokens": cfg.train_batch_tokens,
            "train_micro_batch_size": cfg.train_micro_batch_size,
            "val_tokens": cfg.val_tokens,
            "val_micro_batch_size": cfg.val_micro_batch_size,
            "lr_scheduler": cfg.lr_scheduler,
            "optimizer": {
                "weight_decay": cfg.weight_decay,
                "grad_clip_norm": cfg.grad_clip_norm,
            },
            "logging": {
                "validation_interval": cfg.validation_interval,
                "sample_interval": cfg.sample_interval,
                "save_interval": cfg.save_interval,
            },
            "sample": cfg.sample,
            "task_evaluation": cfg.task_evaluation,
        },
    }
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, indent=4, sort_keys=False)


def debug_config(cfg: TrainConfig, world_size: int) -> TrainConfig:
    batch_tokens = 2 * cfg.train_micro_batch_size * cfg.context_length * world_size
    return replace(
        cfg,
        train_batch_tokens=batch_tokens,
        train_tokens=2 * batch_tokens,
        val_tokens=cfg.val_micro_batch_size * cfg.context_length * world_size,
        validation_interval=1,
        sample_interval=1,
        save_checkpoints=True,
        sample={**cfg.sample, "n_samples": 1, "max_new_tokens": 16},
    )
