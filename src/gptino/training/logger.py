import json
from pathlib import Path

import torch
import torch.distributed as dist


class LoggerLite:
    def __init__(self, log_dir: Path, logging_interval: int, ddp_context, experiment_name:str | None = None):
        self.master_process = ddp_context.master_process
        self.ddp_enabled = ddp_context.enabled
        self.global_step = 0
        self.logging_interval = logging_interval

        self.log_path = log_dir if experiment_name is None else log_dir / experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.log_dict = {"step": 0}
        self.console_log = {"step": 0}

    def log_value(self, value, name, console_log=True, reduce=True):
        if self.ddp_enabled and reduce:
            dist.all_reduce(value, op=dist.ReduceOp.AVG)
        if not self.master_process or self.global_step % self.logging_interval != 0:
            return
        assert name not in self.log_dict, f"Metric {name} already logged for this step"
        self.log_dict[name] = value.item() if isinstance(value, torch.Tensor) else value
        if console_log:
            self.console_log[name] = value.item() if isinstance(value, torch.Tensor) else value

    def step(self):
        if not self.master_process:
            return
        self._log_metrics()
        self.global_step += 1
        self.log_dict = {"step": self.global_step}
        self.console_log = {"step": self.global_step}

    def _log_metrics(self):
        with open(self.log_path / "metrics.jsonl", "a") as f:
            json.dump(dict(self.log_dict), f)
            f.write("\n")
        print(f"Step {self.global_step} | " + " | ".join(f"{k}: {v:.4f}" for k, v in self.console_log.items() if k != "step"), flush=True)