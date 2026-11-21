import math
from dataclasses import dataclass

import torch


@dataclass
class LRSchedulerConfig:
    warmup_steps: int
    decay_steps: int
    peak_lr: float
    initial_lr: float
    final_lr: float


class LinearWarmupCosineDecayLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, cfg: LRSchedulerConfig, last_epoch=-1):
        self.cfg = cfg
        self.optimizer = optimizer
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        return {k: v for k, v in super().state_dict().items() if k != "cfg"}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

    def get_lr(self) -> list[float | torch.Tensor]:
        step = self.last_epoch
        lr = self.compute_lr(step)
        return [lr for _ in self.optimizer.param_groups]

    def compute_lr(self, step):
        if step < self.cfg.warmup_steps:
            return (self.cfg.peak_lr - self.cfg.initial_lr) * (
                step / self.cfg.warmup_steps
            ) + self.cfg.initial_lr
        if step > self.cfg.decay_steps:
            return self.cfg.final_lr
        decay_ratio = (step - self.cfg.warmup_steps) / (
            self.cfg.decay_steps - self.cfg.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.cfg.final_lr + coeff * (self.cfg.peak_lr - self.cfg.final_lr)
