import math
from dataclasses import dataclass

import torch


@dataclass
class LRSchedulerConfig:
    warmup_steps: int
    decay_steps: int
    min_lr: float
    max_lr: float

class LinearWarmupCosineDecayLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, cfg: LRSchedulerConfig, last_epoch=-1):
        self.optimizer = optimizer
        self.cfg = cfg
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lr = self.compute_lr(step)
        return [lr for _ in self.optimizer.param_groups]

    def compute_lr(self, step):
            if step < self.cfg.warmup_steps:
                return (self.cfg.max_lr - self.cfg.min_lr) * (step / self.cfg.warmup_steps) + self.cfg.min_lr
            if step > self.cfg.decay_steps:
                return self.cfg.min_lr
            decay_ratio = (step - self.cfg.warmup_steps) / (self.cfg.decay_steps - self.cfg.warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.cfg.min_lr + coeff * (self.cfg.max_lr - self.cfg.min_lr)