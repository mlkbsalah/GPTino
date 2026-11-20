import contextlib
import time

import torch
import torch.distributed as dist

from .callbacks import TrainSummaryCallback
from .rng import set_rng_state


@contextlib.contextmanager
def profile_step(device, tokens, ddp_context):
    device_type = torch.device(device).type

    def synchronize():
        if device_type == "cuda":
            torch.cuda.synchronize(device)
        elif device_type == "mps":
            torch.mps.synchronize()

    metrics = {}
    synchronize()
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    yield metrics
    synchronize()
    elapsed = time.perf_counter() - start

    stats = [elapsed]
    if device_type == "cuda":
        stats += [torch.cuda.max_memory_allocated(device) / 1024**3]
    if device_type == "mps":
        stats += [torch.mps.max_memory_allocated(device) / 1024**3]

    stats = torch.tensor(stats, device=device)
    if ddp_context.enabled:
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        tokens *= ddp_context.world_size
    metrics["Training Tokens/sec"] = tokens / stats[0]
    metrics["peak_gpu"] = stats[1] if device_type == "cuda" else stats[2]


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
        resume,
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

        self.step = 0  # step stores the number of training steps taken
        self.raw_model = model if not self.ddp else model.module

        self.resume = resume

    def load_training_checkpoint(self, resume):
        checkpoint = torch.load(resume, map_location=self.device, weights_only=False)
        rng_states = checkpoint.get("rng_states")
        if len(rng_states) != self.ddp_context.world_size:
            raise ValueError(
                "RNG resume requires the same world size as the checkpoint"
            )
        self.raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["completed_steps"]
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        return None if rng_states is None else rng_states[self.ddp_context.rank]

    def _run_callbacks(self, event_name, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event_name)(self, **kwargs)

    def fit(self, train_dataloader, validation_dataloader):
        rng_state = None
        if self.resume is not None:
            rng_state = self.load_training_checkpoint(self.resume)
        if rng_state is not None:
            set_rng_state(rng_state, self.device)

        self._run_callbacks("on_train_begin")
        for batch in train_dataloader.iter_from(self.step):
            self._run_callbacks("on_step_begin")

            self.model.train()
            with profile_step(self.device, batch[0].numel(), self.ddp_context) as profile_metrics:
                batch_loss = self.train_batch(batch)
            self.record_metric_dict(profile_metrics, reduce=False)
            self.step += 1

            self.record_metric("Training Loss", batch_loss, reduce=True)

            if (self.step % self.validation_interval == 0) or (
                self.step == len(train_dataloader)
            ):
                self._run_callbacks("on_validation_begin")

                self.model.eval()
                val_loss = self.evaluate(validation_dataloader)

                self.record_metric("Validation Loss", val_loss, reduce=True)
                self._run_callbacks("on_validation_end")

            self._run_callbacks("on_step_end")
            self.record_metric("Step", self.step, reduce=False)
            self.flush_metrics()

        self._run_callbacks("on_train_end")

    def record_metric(self, name, value, reduce):
        if self.ddp and reduce:
            dist.all_reduce(value, op=dist.ReduceOp.AVG)
        if self.ddp_context.master_process:
            self.logger.record(name, value)

    def record_metric_dict(self, metric_dict, reduce):
        for name, value in metric_dict.items():
            self.record_metric(name, value, reduce)

    def flush_metrics(self):
        if self.ddp_context.master_process:
            self.logger.flush()

    def train_batch(self, batch):
        self.optimizer.zero_grad()
        loss_accum = torch.zeros(1, device=self.device)
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
        self.record_metric("Gradient Norm", norm, reduce=False)
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss_accum

    @torch.inference_mode()
    def evaluate(self, val_dataloader):
        val_loss_accum = torch.zeros(1, device=self.device)
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
        yield from zip(
            x.split(self.micro_batch_size), y.split(self.micro_batch_size)
        )  # split sequences since i want context lenght to stay the same.
