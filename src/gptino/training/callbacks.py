from pathlib import Path

import torch


class BaseCallback:
    def on_train_begin(self, trainer):pass
    def on_validation_begin(self, trainer):pass
    def on_validation_end(self, trainer):pass
    def on_step_begin(self, trainer):pass
    def on_step_end(self, trainer):pass
    def on_train_end(self, trainer):pass



class CheckpointCallback(BaseCallback):
    def __init__(self, save_dir: Path, save_interval: int):
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.latest_checkpoint_path = Path(self.save_dir) / "latest.pt"

    def on_step_end(self, trainer):
        if not trainer.ddp_context.master_process:
            return
        if trainer.step % self.save_interval == 0 and trainer.step > 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.save_dir / f"checkpoint_step_{trainer.step}.pt"


            torch.save(
                {
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "lr_scheduler_state_dict": trainer.lr_scheduler.state_dict(),
                    "step": trainer.step,
                },
                checkpoint_path
            )
            self.latest_checkpoint_path.unlink(missing_ok=True)
            self.latest_checkpoint_path.symlink_to(checkpoint_path.name)

class LearningRateCallback(BaseCallback):
    def __init__(self):
        pass

    def on_step_begin(self, trainer):
        current_lr = trainer.lr_scheduler.get_last_lr()[0]
        trainer.logger.log_value(current_lr, "Learning Rate", reduce=False)

class TrainSummaryCallback(BaseCallback):
    def on_train_begin(self, trainer):
        if not trainer.ddp_context.master_process:
            return
        num_trainable_param = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        om_dict = {"M": 1e6, "K": 1e3}
        om = "M" if num_trainable_param > 1e6 else "K" if num_trainable_param > 1e3 else ""
        print(f"Number of trainable parameters: {num_trainable_param/om_dict[om]:.2f}{om}")
        print("Training started.")
