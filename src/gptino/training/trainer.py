import contextlib
from .callbacks import TrainSummaryCallback
import torch


class TrainerLite:
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        ddp_context,
        micro_batch_size,
        validation_interval,
        gradient_clip_norm,
        logger,
        callbacks,
    ):
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.callbacks = [TrainSummaryCallback()] + callbacks
        self.ddp_context = ddp_context
        self.device = ddp_context.device
        self.ddp = ddp_context.enabled
        self.gradient_clip_norm = gradient_clip_norm

        self.validation_interval = validation_interval
        self.micro_batch_size = micro_batch_size

        self.step = 0

    def _run_callbacks(self, event_name, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event_name)(self, **kwargs)

    def fit(self, train_dataloader, validation_dataloader):
        self._run_callbacks("on_train_begin")
        max_steps = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            self.step = step
            if (step % self.validation_interval == 0) or (step == max_steps - 1):
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
        loss_accum = 0.0
        micro_batches = list(self.micro_batch_generator(batch))
        num_sequences = batch[0].size(0)
        for micro_step, micro_batch in enumerate(micro_batches):
            ctx = (
                self.model.no_sync()
                if self.ddp and micro_step < (len(micro_batches) - 1)
                else contextlib.nullcontext()
            )
            with ctx:
                loss = self.compute_loss(micro_batch)
                loss *= micro_batch[0].size(0) / num_sequences
                loss_accum += loss.detach()
                loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.gradient_clip_norm
        )
        self.logger.log_value(norm, "Gradient Norm")
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss_accum

    @torch.no_grad()
    def evaluate(self, val_dataloader):
        val_loss_accum = 0.0
        for batch in val_dataloader:
            loss = self.compute_loss(batch)
            val_loss_accum += loss.detach()
        val_loss_accum /= len(val_dataloader)
        return val_loss_accum

    def compute_loss(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        with torch.autocast(
            device_type=torch.device(self.device).type, dtype=torch.bfloat16
        ):
            logits = self.model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )  # mean reduction over all tokens in the batch reads as the error per token
        return loss

    def micro_batch_generator(self, batch):
        x, y = batch
        yield from zip(x.split(self.micro_batch_size), y.split(self.micro_batch_size))
