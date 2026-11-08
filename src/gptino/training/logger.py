import json
from pathlib import Path

import torch


class LoggerLite:
    def __init__(
        self,
        log_dir: Path,
        logging_interval: int,
        ddp_context,
        experiment_name: str | None = None,
    ):
        self.master_process = ddp_context.master_process
        self.ddp_enabled = ddp_context.enabled
        self.logging_interval = logging_interval

        self.log_path = (
            log_dir if experiment_name is None else log_dir / experiment_name
        )
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.log_dict: dict[str, float] = {}
        self.console_log: dict[str, float] = {}

    def log_value(self, value, name, log_on_console=True):
        assert name not in self.log_dict, f"Metric {name} already logged for this step"
        self.log_dict[name] = value.item() if isinstance(value, torch.Tensor) else value
        if log_on_console:
            self.console_log[name] = (
                value.item() if isinstance(value, torch.Tensor) else value
            )

    def write_metrics(self):
        with open(self.log_path / "metrics.jsonl", "a") as f:
            json.dump(dict(self.log_dict), f)
            f.write("\n")
        print(
            " | ".join(
                f"{k}: {v:.4f}" for k, v in self.console_log.items() if k != "step"
            ),
            flush=True,
        )

    def flush(self):
        self.log_dict.clear()
        self.console_log.clear()
