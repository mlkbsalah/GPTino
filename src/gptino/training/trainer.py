import contextlib
from dataclasses import dataclass
from .callbacks import TrainSummaryCallback
import torch


@dataclass
class Budget:
    num_val_tokens: int
    num_train_tokens: int
    train_batch_size: int
    train_micro_batch_size: int
    val_micro_batch_size: int
    context_length: int
    world_size: int

    def __post_init__(self):
        self.train_tokens_per_micro_step = self.context_length * self.train_micro_batch_size * self.world_size
        self.val_tokens_per_micro_step = self.context_length * self.val_micro_batch_size * self.world_size

        self.grad_accum_steps = self.train_batch_size // self.train_tokens_per_micro_step        
        self.train_max_steps = self.num_train_tokens // self.train_batch_size
        self.val_max_steps = self.num_val_tokens // self.val_tokens_per_micro_step

        assert self.train_batch_size % self.train_tokens_per_micro_step == 0, \
            f"Train batch size {self.train_batch_size} must be divisible by tokens per micro step {self.train_tokens_per_micro_step}"
        assert (self.num_val_tokens % self.val_tokens_per_micro_step == 0), \
            f"Validation tokens {self.num_val_tokens} must be divisible by tokens per micro step {self.val_tokens_per_micro_step}"



class TrainerLite:
    def __init__(self, model, optimizer, lr_scheduler, ddp_context, budget, validation_interval, gradient_clip_norm, logger, callbacks):
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.callbacks = [TrainSummaryCallback()] + callbacks
        self.device = ddp_context.device
        self.ddp = ddp_context.enabled
        self.gradient_clip_norm = gradient_clip_norm

        self.validation_interval = validation_interval

        self.train_max_steps = budget.train_max_steps
        self.val_max_steps = budget.val_max_steps
        self.grad_accum_steps = budget.grad_accum_steps
        self.train_micro_batch_size = budget.train_micro_batch_size
        self.val_micro_batch_size = budget.val_micro_batch_size

        self.step = 0


    def _run_callbacks(self, event_name, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event_name)(self, **kwargs)



    def fit(self, train_dataloader, validation_dataloader):
        self._run_callbacks("on_train_begin")
        for step, batch in enumerate(train_dataloader):
            self.step = step
            if step >= self.train_max_steps:
                break
            if (step % self.validation_interval == 0) or (step == self.train_max_steps - 1):
                self._run_callbacks("on_validation_begin")
                self.model.eval()
                val_loss = self.evaluate(validation_dataloader)
                self.logger.log_value(val_loss, "Validation Loss", reduce=True)
                self._run_callbacks("on_validation_end")

            self._run_callbacks("on_step_begin")
            self.model.train()
            batch_loss = self.train_batch(batch)
            self._run_callbacks("on_step_end")

            self.logger.log_value(batch_loss, "Training Loss", reduce=True)
            self.logger.step()

        self._run_callbacks("on_train_end")


    def train_batch(self, batch):
        self.optimizer.zero_grad()
        loss_accum = 0.
        for micro_step, micro_batch in enumerate(self.micro_batch_generator(batch)):
            ctx = self.model.no_sync() if self.ddp and micro_step < (self.grad_accum_steps - 1) else contextlib.nullcontext()
            with ctx:
                loss = self.compute_loss(micro_batch)
                loss /= self.grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)            
        self.logger.log_value(norm, "Gradient Norm")
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss_accum


    @torch.no_grad()
    def evaluate(self, val_dataloader):
        val_loss_accum = 0.
        for val_step, batch in enumerate(val_dataloader):
            if val_step >= self.val_max_steps:
                break
            loss = self.compute_loss(batch)
            val_loss_accum += loss.detach()
        val_loss_accum /= len(val_dataloader)
        return val_loss_accum

    
    def compute_loss(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16):
            logits = self.model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss

    def micro_batch_generator(self, batch):
        x, y = batch
        yield from zip(x.split(self.train_micro_batch_size), y.split(self.train_micro_batch_size))

        

