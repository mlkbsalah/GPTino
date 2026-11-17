from types import SimpleNamespace

import torch
from torch.nn import functional as F

from gptino.module.gpt import GPT, GPTConfig
from gptino.training.callbacks import CheckpointCallback
from gptino.training.lr_scheduler import LinearWarmupCosineDecayLR, LRSchedulerConfig


def test_checkpoint_restores_saved_training_state(tmp_path):
    torch.manual_seed(42)
    config = GPTConfig(context_length=4, vocab_size=16, n_layer=1, n_head=2, n_embd=8)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler_config = LRSchedulerConfig(
        warmup_steps=2, decay_steps=10, min_lr=0.001, max_lr=0.01
    )
    scheduler = LinearWarmupCosineDecayLR(optimizer, scheduler_config)
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, 4]])

    loss = F.cross_entropy(model(x).view(-1, 16), y.view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()

    trainer = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        step=1,
        ddp_context=SimpleNamespace(master_process=True),
    )
    callback = CheckpointCallback(tmp_path / "checkpoints", save_interval=2)
    callback.on_train_begin(trainer)
    callback.on_step_end(trainer)
    assert not (callback.save_dir / "checkpoint_step_1.pt").exists()

    # Finishing training saves all the time
    callback.on_train_end(trainer)
    assert (
        callback.latest_checkpoint_path.resolve()
        == callback.save_dir / "checkpoint_step_1.pt"
    )

    # This file was created by this test; it contains the GPTConfig dataclass.
    checkpoint = torch.load(callback.latest_checkpoint_path, weights_only=False)
    restored_model = GPT(checkpoint["model_config"])
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_scheduler = LinearWarmupCosineDecayLR(restored_optimizer, scheduler_config)
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restored_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    assert checkpoint["completed_steps"] == 1
    assert restored_model.config == config
    torch.testing.assert_close(restored_model(x), model(x))
    torch.testing.assert_close(restored_optimizer.state_dict(), optimizer.state_dict())
    assert restored_scheduler.state_dict() == scheduler.state_dict()

    trainer.step = 2
    callback.on_step_end(trainer)
    assert (callback.save_dir / "checkpoint_step_2.pt").exists()
    assert (
        callback.latest_checkpoint_path.resolve()
        == callback.save_dir / "checkpoint_step_2.pt"
    )
