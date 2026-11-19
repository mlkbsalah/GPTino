from pathlib import Path

import tiktoken
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from gptino.config import (
    load_train_config,
    save_config,
    setup_ddp,
)
from gptino.data.dataloader import DataLoaderLite
from gptino.data.hellaswag import HellaSwagDataset
from gptino.data.piqa import PIQADataset
from gptino.evaluation.multiple_choice_evaluator import MultipleChoiceQuestionEvaluator
from gptino.module.gpt import GPT, GPTConfig
from gptino.training.callbacks import (
    CheckpointCallback,
    CompleteTextCallback,
    DownStreamEvaluationCallback,
    LearningRateCallback,
)
from gptino.training.logger import LoggerLite
from gptino.training.lr_scheduler import LinearWarmupCosineDecayLR, LRSchedulerConfig
from gptino.training.rng import seed_everything
from gptino.training.trainer import TrainerLite

CONFIG_PATH = "config/gpt_config.yaml"


def build_downstream_callbacks(config, ddp_context, save_path):
    if not config:
        return []

    dataloaders = {}
    for name, dataset_class in (
        ("hellaswag", HellaSwagDataset),
        ("piqa", PIQADataset),
    ):
        path = config.get(f"{name}_path")
        if path is None:
            continue
        dataset = dataset_class(path)
        dataloaders[name] = DataLoader(
            dataset,
            batch_size=None,
            sampler=range(ddp_context.rank, len(dataset), ddp_context.world_size),
        )
    evaluators = [MultipleChoiceQuestionEvaluator(dataloaders, ddp_context)]

    return [DownStreamEvaluationCallback(config["interval"], evaluators, save_path)]


def main(config_path=CONFIG_PATH, run_dir=Path("runs"), debug=False, resume=None):
    assert not (debug and resume is not None), (
        "Cannot debug and resume from a checkpoint at the same time"
    )
    run_dir = Path(resume).parent.parent if resume is not None else run_dir
    logger = LoggerLite(run_dir)
    config_path = run_dir / "config.yaml" if resume else config_path
    ddp_context = setup_ddp()
    cfg = load_train_config(config_path, debug=debug, world_size=ddp_context.world_size)
    if resume is not None:
        print(f"Resuming from {resume}, setting config to {config_path}")
    if ddp_context.master_process:
        save_config(cfg, run_dir / "config.yaml")

    seed_everything(cfg.seed, ddp_context.device)

    train_loader = DataLoaderLite(
        data_dir=cfg.data_dir,
        max_tokens=cfg.train_tokens,
        split="train",
        batch_tokens=cfg.train_batch_tokens,
        context_length=cfg.context_length,
        process_rank=ddp_context.rank,
        world_size=ddp_context.world_size,
        master_process=ddp_context.master_process,
    )

    val_loader = DataLoaderLite(
        data_dir=cfg.data_dir,
        max_tokens=cfg.val_tokens,
        split="val",
        batch_tokens=cfg.val_micro_batch_size
        * cfg.context_length
        * ddp_context.world_size,
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
        params=[
            {
                "params": [p for p in model.parameters() if p.dim() >= 2],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for p in model.parameters() if p.dim() <= 1],
                "weight_decay": 0.0,
            },
        ],
        lr=cfg.lr_scheduler["min_lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )

    max_steps = len(train_loader)
    lr_scheduler = LinearWarmupCosineDecayLR(
        optimizer=optimizer,
        cfg=LRSchedulerConfig(
            warmup_steps=int(cfg.lr_scheduler["warmup_fraction"] * max_steps),
            decay_steps=max_steps,
            min_lr=cfg.lr_scheduler["min_lr"],
            max_lr=cfg.lr_scheduler["max_lr"],
        ),
    )

    trainer = TrainerLite(
        model=model,
        micro_batch_size=cfg.train_micro_batch_size,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        ddp_context=ddp_context,
        validation_interval=cfg.validation_interval,
        gradient_clip_norm=cfg.grad_clip_norm,
        callbacks=[
            CheckpointCallback(
                save_dir=run_dir / "checkpoints", save_interval=cfg.save_interval
            ),
            CompleteTextCallback(
                tokenizer=tiktoken.get_encoding("gpt2"),
                save_path=run_dir / "sampler.txt",
                sample_every_n_steps=cfg.sample_interval,
                seed=cfg.seed,
                **cfg.sample,
            ),
            LearningRateCallback(),
        ]
        + build_downstream_callbacks(
            cfg.task_evaluation, ddp_context, save_path=run_dir / "task_eval.jsonl"
        ),
        resume=resume,
    )

    trainer.fit(train_loader, val_loader)

    if ddp_context.enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=Path,
        default="runs/debug/",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run two optimizer steps with validation, sampling, and logging every step",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    if args.resume is not None and args.debug:
        raise ValueError("Cannot resume from a checkpoint while in debug mode")
    main(run_dir=args.run_dir, debug=args.debug, resume=args.resume)
