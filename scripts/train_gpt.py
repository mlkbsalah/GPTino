
from pathlib import Path

import tiktoken
import torch
import torch.distributed as dist
from gptino.data.hellaswag import HellaSwagDataset
from gptino.data.piqa import PIQADataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from gptino.config import (
    TrainConfig,
    load_train_config,
    save_config,
)
from gptino.data.dataloader import DataLoaderLite
from gptino.evaluation.multiple_choice_evaluator import MultipleChoiceQuestionEvaluator
from gptino.module.gpt import GPT, GPTConfig
from gptino.parallel_config import DDPContext, setup_ddp
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


def train(cfg: TrainConfig, run_dir: Path, ddp_context: DDPContext, resume: Path | None = None):
    logger = LoggerLite(run_dir)

    seed_everything(cfg.seed, ddp_context.device)

    train_loader = DataLoaderLite(
        data_dir=cfg.data_dir,
        max_tokens=cfg.run_token_budget if cfg.run_token_budget is not None else cfg.train_tokens,
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
    use_cuda = torch.device(ddp_context.device).type == "cuda"
    if ddp_context.enabled:
        model = DDP(model, device_ids=[ddp_context.local_rank])
    if use_cuda:
        model.compile()  # In-place: checkpoint keys don't gain an _orig_mod prefix.

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
        lr=cfg.lr_scheduler["initial_lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True if use_cuda else None,
        weight_decay=cfg.weight_decay,
    )

    schedule_steps = cfg.train_tokens // cfg.train_batch_tokens
    lr_scheduler = LinearWarmupCosineDecayLR(
        optimizer=optimizer,
        cfg=LRSchedulerConfig(
            warmup_steps=(cfg.lr_scheduler["warmup_tokens"] + cfg.train_batch_tokens - 1)
            // cfg.train_batch_tokens,
            decay_steps=schedule_steps,
            peak_lr=cfg.lr_scheduler["peak_lr"],
            initial_lr=cfg.lr_scheduler["initial_lr"],
            final_lr=cfg.lr_scheduler["final_lr_fraction"] * cfg.lr_scheduler["peak_lr"],
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


def main(config_path=CONFIG_PATH, run_dir=Path("runs"), debug=False, resume=None, overrides=()):
    if resume is not None:
        if debug:
            raise ValueError("Cannot resume from a checkpoint while in debug mode")
        run_dir = Path(resume).parent.parent
        config_path = run_dir / "config.yaml"
    ddp_context = setup_ddp()
    try:
        cfg = load_train_config(
            config_path, debug=debug, world_size=ddp_context.world_size,
            overrides=overrides, resume=resume is not None,
        )
        if ddp_context.enabled:
            dist.barrier() # very important since fast processing can save the new config while another may be reading it, only problemitc in resume because processes read and write the same file.
        if ddp_context.master_process:
            run_dir.mkdir(parents=True, exist_ok=True)
            if resume is not None:
                print(f"Resuming from {resume} with config {config_path}")
            save_config(cfg, run_dir / "config.yaml")
        train(cfg, run_dir, ddp_context=ddp_context, resume=resume)
    finally:
        if ddp_context.enabled:
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

    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config field",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CONFIG_PATH),
        help="Path to config file",
    )
    args = parser.parse_args()

    main(config_path=args.config, run_dir=args.run_dir, debug=args.debug, resume=args.resume, overrides=args.overrides)
