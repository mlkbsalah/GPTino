from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gptino.data.dataloader import DataLoaderLite
from gptino.module.gpt import GPT, GPTConfig
from gptino.training.callbacks import CheckpointCallback, LearningRateCallback
from gptino.training.logger import LoggerLite
from gptino.training.lr_scheduler import LinearWarmupCosineDecayLR, LRSchedulerConfig
from gptino.training.trainer import Budget, TrainerLite

CONFIG_PATH = "config/gpt2_train_config.yaml"


@dataclass
class TrainConfig:
    data_dir: str
    model: dict
    seed: int
    train_micro_batch_size: int
    val_micro_batch_size: int
    context_length: int
    train_batch_size: int
    train_tokens: int
    val_tokens: int
    # max_steps: int
    # val_loss_steps: int
    
    lr_scheduler: dict
    
    weight_decay: float
    grad_clip_norm: float
    
    validation_interval: int
    logging_interval: int
    sample_interval: int
    save_interval: int
    save_checkpoints: bool
    
    sample_prompt: str
    sample_num_sequences: int
    sample_max_length: int
    sample_top_k: int
    
@dataclass
class DDPContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    master_process: bool
    device: str

def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)

def load_train_config(path, debug, world_size):
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
        train_batch_size=training["train_batch_size"],
        train_tokens=int(training["train_tokens"]),
        val_tokens=int(training["val_tokens"]),
        
        lr_scheduler=lr,

        weight_decay=optimizer["weight_decay"],
        grad_clip_norm=optimizer["grad_clip_norm"],

        validation_interval=logging["validation_interval"],
        logging_interval=logging["logging_interval"],
        sample_interval=logging["sample_interval"],
        save_interval=logging["save_interval"],
        save_checkpoints=True,

        sample_prompt=sample["prompt"],
        sample_num_sequences=sample["num_return_sequences"],
        sample_max_length=sample["max_length"],
        sample_top_k=sample["top_k"],
    )

    return cfg

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    return device

def seed_everything(seed, device):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif device == "mps":
        torch.mps.manual_seed(seed)
        
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


def main(debug=False, config_path=CONFIG_PATH, run_dir=Path("runs")):

    ddp_context = setup_ddp()
    cfg = load_train_config(config_path, debug, ddp_context.world_size)
    seed_everything(cfg.seed, ddp_context.device)

    budget = Budget(
        num_val_tokens=cfg.val_tokens,
        num_train_tokens=cfg.train_tokens,
        train_batch_size=cfg.train_batch_size,
        train_micro_batch_size=cfg.train_micro_batch_size,
        val_micro_batch_size=cfg.val_micro_batch_size,
        context_length=cfg.context_length,
        world_size=ddp_context.world_size,
    )

    train_loader = DataLoaderLite(
        data_dir=cfg.data_dir,
        max_tokens=cfg.train_tokens,
        split="train",
        batch_size_tokens=cfg.train_batch_size // ddp_context.world_size,
        context_length=cfg.context_length,
        process_rank=ddp_context.rank,
        world_size=ddp_context.world_size,
        master_process=ddp_context.master_process,
    )


    val_loader = DataLoaderLite(
        data_dir=cfg.data_dir,
        max_tokens=cfg.val_tokens,
        split="val",
        batch_size_tokens=cfg.val_micro_batch_size * cfg.context_length,
        context_length=cfg.context_length,
        process_rank=ddp_context.rank,
        world_size=ddp_context.world_size,
        master_process=ddp_context.master_process,
    )

    model = GPT(GPTConfig(**cfg.model))
    model.to(ddp_context.device)
    if ddp_context.enabled:
        model = DDP(model, device_ids=[ddp_context.local_rank])



    optimizer = torch.optim.AdamW(
        params=[{"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": cfg.weight_decay},
                {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0}],
        lr=cfg.lr_scheduler["min_lr"], # will be overridden by the scheduler, but we need to set it to something
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )

    lr_scheduler = LinearWarmupCosineDecayLR(
        optimizer=optimizer,
        cfg=LRSchedulerConfig(**cfg.lr_scheduler),
    )

    logger = LoggerLite(
        log_dir=run_dir,
        ddp_context=ddp_context,
        logging_interval=cfg.logging_interval
    )
    
    trainer = TrainerLite(
        model=model,
        budget=budget,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        ddp_context=ddp_context,
        validation_interval=cfg.validation_interval,
        gradient_clip_norm=cfg.grad_clip_norm,
        callbacks=[
            CheckpointCallback(save_dir=run_dir / "checkpoints", save_interval=cfg.save_interval),
            LearningRateCallback(),
        ]
    )

    trainer.fit(train_loader, val_loader)

    if ddp_context.enabled:
        dist.barrier()
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",
                        type=Path,
                        default="runs/debug/",
                        help="Directory to save logs and checkpoints")
    args = parser.parse_args()
    main(run_dir=args.run_dir)