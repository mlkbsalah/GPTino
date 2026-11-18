from copy import deepcopy
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from gptino.training.logger import LoggerLite
from gptino.training.trainer import TrainerLite


def test_accumulated_update_matches_full_batch(tmp_path):
    torch.manual_seed(42)
    full_model = torch.nn.Embedding(8, 8)
    accumulated_model = deepcopy(full_model)
    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(accumulated_optimizer, step_size=10)
    context = SimpleNamespace(enabled=False, device="cpu", master_process=True)
    trainer = TrainerLite(
        model=accumulated_model,
        optimizer=accumulated_optimizer,
        lr_scheduler=scheduler,
        ddp_context=context,
        micro_batch_size=2,
        validation_interval=1,
        gradient_clip_norm=1.0,
        logger=LoggerLite(tmp_path),
        callbacks=[],
    )
    x = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [0, 2]])
    y = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 0], [2, 4]])

    # Five sequences split into 2 + 2 + 1 exercises the smaller final micro-batch.
    chunks = list(trainer.micro_batch_generator((x, y)))
    assert [chunk_x.size(0) for chunk_x, _ in chunks] == [2, 2, 1]

    full_optimizer.zero_grad()
    full_loss = F.cross_entropy(full_model(x).view(-1, 8), y.view(-1))
    full_loss.backward()
    torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=1.0)
    full_optimizer.step()

    accumulated_loss = trainer.train_batch((x, y))

    torch.testing.assert_close(accumulated_loss.squeeze(), full_loss.detach())
    for full_parameter, accumulated_parameter in zip(
        full_model.parameters(), accumulated_model.parameters()
    ):
        torch.testing.assert_close(accumulated_parameter.grad, full_parameter.grad)
        torch.testing.assert_close(accumulated_parameter, full_parameter)
    assert scheduler.last_epoch == 1
