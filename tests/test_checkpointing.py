import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from gptino.module.gpt import GPT, GPTConfig
from gptino.training.callbacks import CheckpointCallback
from gptino.training.lr_scheduler import LinearWarmupCosineDecayLR, LRSchedulerConfig
from gptino.training.rng import get_rng_state, seed_everything, set_rng_state
from gptino.training.trainer import TrainerLite


def test_rng_state_restores_all_cpu_streams():
    original = get_rng_state("cpu")
    try:
        seed_everything(42, "cpu")
        saved = get_rng_state("cpu")
        expected = (random.random(), np.random.rand(), torch.rand(4))
        seed_everything(99, "cpu")
        set_rng_state(saved, "cpu")
        assert random.random() == expected[0]
        assert np.random.rand() == expected[1]
        torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
        seed_everything(42, "cpu")
        assert random.random() == expected[0]
        assert np.random.rand() == expected[1]
        torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
    finally:
        set_rng_state(original, "cpu")


@pytest.mark.parametrize("event", ["on_step_end", "on_train_end"])
def test_non_master_participates_in_rng_gather(tmp_path, monkeypatch, event):
    gather = Mock()
    monkeypatch.setattr("gptino.training.callbacks.dist.all_gather_object", gather)
    trainer = SimpleNamespace(
        device="cpu",
        ddp=True,
        step=2,
        ddp_context=SimpleNamespace(master_process=False, world_size=2, rank=1),
    )
    callback = CheckpointCallback(tmp_path, save_interval=2)
    getattr(callback, event)(trainer)
    gather.assert_called_once()
    assert len(gather.call_args.args[0]) == 2
    assert "torch" in gather.call_args.args[1]
    assert not list(tmp_path.iterdir())


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
        device="cpu",
        ddp=False,
        ddp_context=SimpleNamespace(master_process=True, world_size=1, rank=0),
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
    expected_random = (random.random(), np.random.rand(), torch.rand(4))
    checkpoint = torch.load(callback.latest_checkpoint_path, weights_only=False)
    set_rng_state(checkpoint["rng_states"][0], "cpu")
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(4), expected_random[2], rtol=0, atol=0)
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

    restored_trainer = TrainerLite(
        model=restored_model,
        optimizer=restored_optimizer,
        lr_scheduler=restored_scheduler,
        ddp_context=SimpleNamespace(
            enabled=False, device="cpu", world_size=1, rank=0, master_process=False
        ),
        micro_batch_size=1,
        validation_interval=1,
        gradient_clip_norm=1.0,
        logger=Mock(),
        callbacks=[],
        resume=callback.latest_checkpoint_path,
    )
    # Startup callbacks may consume randomness; restore after they run.
    startup = Mock()
    startup.on_train_begin.side_effect = lambda trainer: torch.rand(10)
    restored_trainer.callbacks = [startup]
    loader = Mock()
    loader.iter_from.return_value = iter(())
    restored_trainer.fit(loader, None)
    loader.iter_from.assert_called_once_with(1)
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(4), expected_random[2], rtol=0, atol=0)

    restored_trainer.ddp_context.world_size = 2
    with pytest.raises(ValueError, match="world size"):
        restored_trainer.load_training_checkpoint(callback.latest_checkpoint_path)

    trainer.step = 2
    callback.on_step_end(trainer)
    assert (callback.save_dir / "checkpoint_step_2.pt").exists()
    assert (
        callback.latest_checkpoint_path.resolve()
        == callback.save_dir / "checkpoint_step_2.pt"
    )
