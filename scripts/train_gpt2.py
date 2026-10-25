from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime

import tiktoken
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from gptino.data.dataloader import DataLoaderLite
from gptino.module.gpt import GPT, GPTConfig

CONFIG_PATH = "/Users/malek/repos/GPTino/config/gpt2_train_config.json"


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
    log_interval: int
    sample_interval: int
    save_interval: int
    save_checkpoints: bool
    sample_prompt: str
    sample_num_sequences: int
    sample_max_length: int
    sample_top_k: int


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


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
        log_interval=logging["log_interval"],
        sample_interval=logging["sample_interval"],
        save_interval=logging["save_interval"],
        save_checkpoints=True,
        sample_prompt=sample["prompt"],
        sample_num_sequences=sample["num_return_sequences"],
        sample_max_length=sample["max_length"],
        sample_top_k=sample["top_k"],
    )

    if debug:
        debug_cfg = raw["debug"]
        cfg.max_steps = debug_cfg["max_steps"]
        cfg.val_loss_steps = debug_cfg["val_loss_steps"]
        cfg.log_interval = debug_cfg["log_interval"]
        cfg.sample_interval = debug_cfg["sample_interval"]
        cfg.total_batch_size = cfg.micro_batch_size * cfg.seq_len * world_size
        cfg.save_checkpoints = False

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
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device


def main(debug=False):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_ddp()
    cfg = load_train_config(CONFIG_PATH, debug, ddp_world_size)
    if master_process:
        print(f"Running on {ddp_world_size} processes. DDP: {ddp}")

    seed_everything(cfg.seed, device)

    enc = tiktoken.get_encoding("gpt2")

    B = cfg.micro_batch_size
    T = cfg.seq_len
    grad_accum_steps = cfg.total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(
            f"total batch size: {cfg.total_batch_size} | micro batch size: {B} | grad accum steps: {grad_accum_steps}"
        )

    train_loader = DataLoaderLite(
        data_dir=cfg.data_dir,
        split="train",
        B=B,
        T=T,
        process_rank=ddp_rank,
        world_size=ddp_world_size,
        master_process=master_process,
    )
    val_loader = DataLoaderLite(
        data_dir=cfg.data_dir,
        split="val",
        B=B,
        T=T,
        process_rank=ddp_rank,
        world_size=ddp_world_size,
        master_process=master_process,
    )

    torch.set_float32_matmul_precision("high")

    model = GPT(GPTConfig(**cfg.model))
    model.to(device)
    if device == "cuda":
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    def get_lr(step):
        if step < cfg.warmup_steps:
            return cfg.max_lr * ((step + 1) / cfg.warmup_steps)
        if step > cfg.decay_steps:
            return cfg.min_lr
        decay_ratio = (step - cfg.warmup_steps) / (cfg.decay_steps - cfg.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.max_lr - cfg.min_lr)

    optimizer = raw_model.configure_optimizer(
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.max_lr,
        device=device,
    )  # type: ignore

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for step in range(cfg.max_steps):
        t0 = time.time()

        # validation
        if step % cfg.log_interval == 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(cfg.val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / cfg.val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"step {step:5d} | val loss: {val_loss_accum:.6f}")

        # sample from the model
        if step > 0 and step % cfg.sample_interval == 0:
            model.eval()
            tokens = enc.encode(cfg.sample_prompt)
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(cfg.sample_num_sequences, 1)
            x = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
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
                print(f"rank {ddp_rank} | sample {i}: {decoded} \n {'-' * 80}")

        # training step
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_second = (
            train_loader.B * train_loader.T * train_loader.world_size
        ) / (t1 - t0)
        if master_process and step % cfg.log_interval == 0:
            print(
                f"step {step:5d} | loss: {loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_second:,.0f}"
            )

        if (
            cfg.save_checkpoints
            and master_process
            and step > 0
            and step % cfg.save_interval == 0
        ):
            torch.save(
                raw_model.state_dict(), f"GPT2-model_step{step}_{run_timestamp}.pt"
            )

    if cfg.save_checkpoints and master_process:
        torch.save(raw_model.state_dict(), f"GPT2-model_{run_timestamp}.pt")

    if ddp:
        dist.barrier()
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="short run with validation and sampling, no checkpoints",
    )
    args = parser.parse_args()
    main(debug=args.debug)
