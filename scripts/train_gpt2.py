from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import tiktoken
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from gptino.data.dataloader import DataLoaderLite
from gptino.module.gpt import GPT, GPTConfig

CONFIG_PATH = "config/gpt2_train_config.yaml"


@dataclass
class TrainConfig:
    data_dir: str
    model: dict
    seed: int
    micro_batch_size: int
    seq_len: int
    total_batch_size: int
    max_steps: int
    val_loss_steps: int
    max_lr: float
    min_lr: float
    warmup_steps: int
    decay_steps: int
    weight_decay: float
    grad_clip_norm: float
    val_log_interval: int
    train_log_interval: int
    sample_interval: int
    save_interval: int
    save_checkpoints: bool
    sample_prompt: str
    sample_num_sequences: int
    sample_max_length: int
    sample_top_k: int
    debug: bool

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

def save_config(cfg: TrainConfig, path):
    with open(path, "w") as f:
        yaml.dump(cfg.__dict__, f, default_flow_style=False)

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
        micro_batch_size=training["micro_batch_size"],
        seq_len=raw["model"]["block_size"],
        total_batch_size=training["total_batch_size"],
        max_steps=training["max_steps"],
        val_loss_steps=training["val_loss_steps"],
        max_lr=lr["max_lr"],
        min_lr=lr["min_lr"],
        warmup_steps=lr["warmup_steps"],
        decay_steps=lr["decay_steps"],
        weight_decay=optimizer["weight_decay"],
        grad_clip_norm=optimizer["grad_clip_norm"],
        val_log_interval=logging["val_log_interval"],
        train_log_interval=logging["train_log_interval"],
        sample_interval=logging["sample_interval"],
        save_interval=logging["save_interval"],
        save_checkpoints=True,
        sample_prompt=sample["prompt"],
        sample_num_sequences=sample["num_return_sequences"],
        sample_max_length=sample["max_length"],
        sample_top_k=sample["top_k"],
        debug=debug,

    )

    if debug:
        debug_cfg = raw["debug"]
        cfg.max_steps = debug_cfg["max_steps"]
        cfg.val_loss_steps = debug_cfg["val_loss_steps"]
        cfg.val_log_interval = debug_cfg["val_log_interval"]
        cfg.train_log_interval = debug_cfg["train_log_interval"]
        cfg.sample_interval = debug_cfg["sample_interval"]
        cfg.total_batch_size = cfg.micro_batch_size * cfg.seq_len * world_size
        cfg.save_checkpoints = False
        cfg.debug = True

    assert (
        cfg.total_batch_size % (cfg.micro_batch_size * cfg.seq_len * world_size) == 0
    ), "total batch size must be divisible by micro batch size and world size"

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

def save_checkpoint(path, raw_model, optimizer, step, cfg, train_loader):
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg.__dict__,
            "loader": {
                "shard": train_loader.current_shard,
                "position": train_loader.current_position,
            },
        },
        path,
    )

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


def main(debug=False, config_path=CONFIG_PATH, run_dir=Path("./unlogged_runs"), max_steps=None):
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    val_metrics_path = run_dir / "val_metrics.jsonl"
    train_metrics_path = run_dir / "train_metrics.jsonl"
    latest_path = checkpoint_dir / "latest.pt"
    samples_path = run_dir / "samples.txt"

    ddp_context = setup_ddp()

    cfg = load_train_config(config_path, debug, ddp_context.world_size)
    if max_steps is not None:
        cfg.max_steps = max_steps

    seed_everything(cfg.seed, ddp_context.device)


    B = cfg.micro_batch_size
    T = cfg.seq_len
    grad_accum_steps = cfg.total_batch_size // (B * T * ddp_context.world_size)

    if ddp_context.master_process:
        save_config(cfg, run_dir / "train_config.json")
        print(f"Running on {ddp_context.world_size} processes. DDP: {ddp_context.enabled}")
        print(f"total batch size: {cfg.total_batch_size} | micro batch size: {B} | grad accum steps: {grad_accum_steps}")
        if debug:
            print("Running in debug mode. No checkpoints will be saved.")

    train_loader = DataLoaderLite(
        data_dir=cfg.data_dir,
        split="train",
        B=B,
        T=T,
        process_rank=ddp_context.rank,
        world_size=ddp_context.world_size,
        master_process=ddp_context.master_process,
    )
    val_loader = DataLoaderLite(
        data_dir=cfg.data_dir,
        split="val",
        B=B,
        T=T,
        process_rank=ddp_context.rank,
        world_size=ddp_context.world_size,
        master_process=ddp_context.master_process,
    )


    torch.set_float32_matmul_precision("high")
    model = GPT(GPTConfig(**cfg.model))
    raw_model = model
    model.to(ddp_context.device)
    if "cuda" in ddp_context.device:
        model = torch.compile(model)
    if ddp_context.enabled:
        model = DDP(model, device_ids=[ddp_context.local_rank])

    # MFU: fwd+bwd FLOPs per token (weights + attention) vs. GPU peak
    n_params = sum(p.numel() for p in raw_model.parameters())
    flops_per_token = 6 * n_params + 12 * cfg.model["n_layer"] * cfg.model["n_embd"] * T
    PEAK_FLOPS = 989e12  # H100 SXM bf16 dense (H100 PCIe: 756e12)

    def get_lr(step):
        if step < cfg.warmup_steps:
            return (cfg.max_lr - cfg.min_lr) * (step / cfg.warmup_steps) + cfg.min_lr
        if step > cfg.decay_steps:
            return cfg.min_lr
        decay_ratio = (step - cfg.warmup_steps) / (cfg.decay_steps - cfg.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.max_lr - cfg.min_lr)

    optimizer = raw_model.configure_optimizer(
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.max_lr,
        device=ddp_context.device,
    )  # type: ignore

    enc = tiktoken.get_encoding("gpt2")
    for step in range(cfg.max_steps):
        # validation
        if (step % cfg.val_log_interval == 0) or (step == cfg.max_steps - 1):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(cfg.val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(ddp_context.device), y.to(ddp_context.device)
                    with torch.autocast(device_type=ddp_context.device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / cfg.val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp_context.enabled:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if ddp_context.master_process:
                with open(val_metrics_path, "a") as f:
                    f.write(json.dumps({"step": step, "val_loss": val_loss_accum.item()}) + "\n")
                print(f"step {step:5d} | val loss: {val_loss_accum.item():.6f}")

        # sample from the model
        if (step % cfg.sample_interval == 0) or (step == cfg.max_steps - 1):
            model.eval()
            tokens = enc.encode(cfg.sample_prompt)
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(cfg.sample_num_sequences, 1)
            x = tokens.to(ddp_context.device)
            sample_rng = torch.Generator(device=ddp_context.device)
            sample_rng.manual_seed(42 + ddp_context.rank)
            while x.size(1) < cfg.sample_max_length:
                with torch.no_grad():
                    logits, loss = model(x)
                    logits = logits[:, -1, :]  # (B, vocab)
                    probs = F.softmax(logits, dim=-1)  # (B, vocab)
                    topk_probs, topk_indices = torch.topk(
                        probs, cfg.sample_top_k, dim=-1
                    )
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B,1)
                    xcol = torch.gather(topk_indices, -1, ix)
                    x = torch.cat((x, xcol), dim=-1)  # (B, length+1)

            for i in range(cfg.sample_num_sequences):
                tokens = x[i].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_context.rank} | sample {i}: {decoded} \n {'-' * 80}")
                if ddp_context.master_process:
                    with open(samples_path, "a") as f:
                        f.write(f" step {step} | rank {ddp_context.rank} | sample {i}: {decoded}\n{'-' * 80}\n")

        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(ddp_context.device), y.to(ddp_context.device)
            with torch.autocast(device_type=ddp_context.device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp_context.enabled:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()
        if ddp_context.enabled:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if "cuda" in ddp_context.device:
            torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_second = (
            train_loader.B * train_loader.T * grad_accum_steps * train_loader.world_size
        ) / (t1 - t0)
        if ddp_context.master_process and step % cfg.train_log_interval == 0:
            on_cuda = "cuda" in ddp_context.device
            mfu = tokens_per_second * flops_per_token / (PEAK_FLOPS * ddp_context.world_size) if on_cuda else None
            mem_gb = torch.cuda.max_memory_allocated(ddp_context.device) / 1e9 if on_cuda else None
            with open(train_metrics_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": loss_accum.item(), "lr": lr, "norm": norm.item(), "dt": dt, "tokens_per_second": tokens_per_second, "mfu": mfu, "peak_mem_gb": mem_gb})+ "\n")
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_second:,.0f}"
                + (f" | mfu: {mfu:.1%} | mem: {mem_gb:.1f}GB" if on_cuda else "")
            )

        if (
            cfg.save_checkpoints
            and ddp_context.master_process
            and step > 0
            and step % cfg.save_interval == 0
        ):
            checkpoint_path = checkpoint_dir / f"step_{step}.pt"
            save_checkpoint(checkpoint_path, raw_model, optimizer, step, cfg, train_loader)
            latest_path.unlink(missing_ok=True)
            latest_path.symlink_to(checkpoint_path.name)
            



    if cfg.save_checkpoints and ddp_context.master_process:
        checkpoint_path = checkpoint_dir / f"step_{cfg.max_steps - 1}.pt"
        save_checkpoint(checkpoint_path, raw_model, optimizer, cfg.max_steps - 1, cfg, train_loader)
        latest_path.unlink(missing_ok=True)
        latest_path.symlink_to(checkpoint_path.name)

    if ddp_context.enabled:
        dist.barrier()
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="short run with validation and sampling, no checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH,
        help="path to the training configuration file",
    )

    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("./unlogged_runs"),
        help="directory to save logs and checkpoints",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        help="maximum number of training steps (overrides config file)",
    )    

    args = parser.parse_args()
    main(debug=args.debug, config_path=args.config, run_dir=args.run_dir, max_steps=args.max_steps)
