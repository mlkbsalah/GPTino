from pathlib import Path

import torch
import torch.distributed as dist

from .rng import get_rng_state


class BaseCallback:
    def on_train_begin(self, trainer):
        pass

    def on_validation_begin(self, trainer):
        pass

    def on_validation_end(self, trainer):
        pass

    def on_step_begin(self, trainer):
        pass

    def on_step_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass


class CheckpointCallback(BaseCallback):
    def __init__(self, save_dir: Path, save_interval: int):
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.latest_checkpoint_path = Path(self.save_dir) / "latest.pt"

    def on_train_begin(self, trainer):
        if trainer.ddp_context.master_process:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer):
        if trainer.step % self.save_interval == 0 and trainer.step > 0:
            self._save_checkpoint(trainer)

    def on_train_end(self, trainer):
        self._save_checkpoint(trainer)

    def _save_checkpoint(self, trainer):
        local_rng_state = get_rng_state(trainer.device)
        if trainer.ddp:
            rng_states = [None] * trainer.ddp_context.world_size
            dist.all_gather_object(rng_states, local_rng_state)
        else:
            rng_states = [local_rng_state]
        if not trainer.ddp_context.master_process:
            return

        module = getattr(trainer.model, "module", trainer.model)
        checkpoint_path = self.save_dir / f"checkpoint_step_{trainer.step}.pt"
        torch.save(
            {
                "model_state_dict": module.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "lr_scheduler_state_dict": trainer.lr_scheduler.state_dict(),
                "model_config": module.config,
                "completed_steps": trainer.step,
                "rng_states": rng_states,
            },
            checkpoint_path,
        )
        self.latest_checkpoint_path.unlink(missing_ok=True)
        self.latest_checkpoint_path.symlink_to(checkpoint_path.name)


class LearningRateCallback(BaseCallback):
    def __init__(self):
        pass

    def on_step_begin(self, trainer):
        current_lr = trainer.lr_scheduler.get_last_lr()[0]
        trainer.record_metric("Learning Rate", current_lr, reduce=False)


class TrainSummaryCallback(BaseCallback):
    def on_train_begin(self, trainer):
        if not trainer.ddp_context.master_process:
            return
        num_trainable_param = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        om_dict = {"M": 1e6, "K": 1e3}
        om = (
            "M"
            if num_trainable_param > 1e6
            else "K"
            if num_trainable_param > 1e3
            else ""
        )
        print(
            f"Number of trainable parameters: {num_trainable_param / om_dict[om]:.2f}{om}"
        )
        print("Training started.")


class CompleteTextCallback(BaseCallback):
    def __init__(
        self,
        sample_text,
        tokenizer,
        max_new_tokens,
        save_path: Path,
        n_samples=1,
        sample_every_n_steps=1,
        seed=1337,
        top_k=10,
    ):
        self.sample_text = sample_text
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.save_path = save_path
        self.n_samples = n_samples
        self.sample_every_n_steps = sample_every_n_steps
        self.top_k = top_k
        self.seed = seed

    def sample(self, trainer):
        rng = torch.Generator(device=trainer.device).manual_seed(
            self.seed + trainer.ddp_context.rank
        )

        model = getattr(trainer.model, "module", trainer.model)
        model.eval()

        tokens = self.tokenizer.encode(self.sample_text)
        start = torch.tensor(tokens, dtype=torch.long, device=trainer.device)
        completions = model.generate(
            start.repeat(self.n_samples, 1),
            max_new_tokens=self.max_new_tokens,
            top_k=self.top_k,
            rng=rng,
        )
        return completions

    def log_completions(self, trainer, completions):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "a", encoding="utf-8") as f:
            for i, completion in enumerate(completions):
                text = self.tokenizer.decode(completion.tolist())
                sample = f"Step {trainer.step} - Rank {trainer.ddp_context.rank} - Sample {i + 1}:\n{text}\n{'-' * 40}"
                print(sample)
                f.write(sample + "\n")

    @torch.no_grad()
    def on_step_end(self, trainer):
        if not trainer.ddp_context.master_process:
            return
        if trainer.step % self.sample_every_n_steps == 0:
            completions = self.sample(trainer)
            self.log_completions(trainer, completions)

    @torch.no_grad()
    def on_train_end(self, trainer):
        if not trainer.ddp_context.master_process:
            return
        completions = self.sample(trainer)
        self.log_completions(trainer, completions)


class DownStreamEvaluationCallback(BaseCallback):
    def __init__(self, evaluation_interval, evaluators):
        self.evaluation_interval = evaluation_interval
        self.evaluators = evaluators

    @torch.inference_mode()
    def on_step_end(self, trainer):
        if trainer.step % self.evaluation_interval != 0:
            return
        model = trainer.raw_model
        for evaluator in self.evaluators:
            result = evaluator.evaluate(model, trainer.device)
            if trainer.ddp_context.master_process:
                for loader, loader_result in result.items():
                    for metric, value in loader_result.items():
                        trainer.record_metric(f"{loader}/{metric}", value, reduce=False)

    @torch.inference_mode()
    def on_train_end(self, trainer):
        self.on_step_end(trainer)
