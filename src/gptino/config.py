from dataclasses import dataclass, replace

import yaml

_RESUME_PATHS = {
    "training.run_token_budget",
}

_INT_PATH = {
    "training.run_token_budget",
}
_FLOAT_PATHS = {
    "training.lr_scheduler.min_lr",
    "training.lr_scheduler.max_lr",
    "training.optimizer.weight_decay",
    "training.optimizer.grad_clip_norm",
}
_NULLABLE_PATHS = {
    "training.run_token_budget",
}

def _apply_override(raw_config, override, resume):
    path, _, text_val = override.partition("=")
    if text_val.strip() == "":
        raise ValueError(f"Empty override: {override}, to set to none override with \"path=none\"")
    elif text_val.strip() == "none" and path not in _NULLABLE_PATHS:
        raise ValueError(f"Only {_NULLABLE_PATHS} can be set to none, not {path}")
    if resume and path not in _RESUME_PATHS:
        raise ValueError(f"Only {_RESUME_PATHS} can be overridden on resume, not {path}")

    *parents, key = path.split(".")
    section = raw_config
    for parent in parents:
        section = section[parent]
    if key not in section:
        raise ValueError(f"Key {key} not found in {path}")

    if path in _INT_PATH:
        expected = int
    elif path in _FLOAT_PATHS:
        expected = float
    else:
        expected = type(section[key])

    value = text_val.strip()
    if value == "none":
        value = None
    elif expected in (int, float, str):
        value = expected(value)
    else:
        raise ValueError(f"Unexpected value type for {path}: {value}")

    section[key] = value



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
    run_token_budget: int | None = None



def load_train_config(path, debug=False, world_size=1, overrides=(), resume=False):
    with open(path) as f:
        raw = yaml.safe_load(f)
    for override in overrides:
        _apply_override(raw, override, resume)
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
        train_tokens=int(raw["data"]["train_tokens"]),
        run_token_budget=training.get("run_token_budget"),
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
        "data": {
            "data_dir": cfg.data_dir,
            "train_tokens": cfg.train_tokens,
        },
        "training": {
            "seed": cfg.seed,
            "run_token_budget": cfg.run_token_budget,
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
        run_token_budget=None,
        val_tokens=cfg.val_micro_batch_size * cfg.context_length * world_size,
        validation_interval=1,
        sample_interval=1,
        save_checkpoints=True,
        sample={**cfg.sample, "n_samples": 1, "max_new_tokens": 16},
    )
