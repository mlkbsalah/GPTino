import numpy as np
import pytest
import torch

from gptino.data.dataloader import DataLoaderLite


def make_loader(data_dir, max_tokens=8, process_rank=0, world_size=1):
    return DataLoaderLite(
        data_dir=data_dir,
        split="train",
        max_tokens=max_tokens,
        batch_tokens=8,
        context_length=4,
        process_rank=process_rank,
        world_size=world_size,
    )


def test_inputs_and_targets_are_shifted_by_one(tmp_path):
    np.save(tmp_path / "train_000.npy", np.arange(9, dtype=np.uint16))

    x, y = next(iter(make_loader(tmp_path)))

    assert torch.equal(x, torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))
    assert torch.equal(y, torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))


def test_ranks_split_each_global_batch(tmp_path):
    np.save(tmp_path / "train_000.npy", np.arange(17, dtype=np.uint16))
    rank_zero = list(make_loader(tmp_path, max_tokens=16, process_rank=0, world_size=2))
    rank_one = list(make_loader(tmp_path, max_tokens=16, process_rank=1, world_size=2))

    for step in range(2):
        x = torch.cat([rank_zero[step][0], rank_one[step][0]])
        y = torch.cat([rank_zero[step][1], rank_one[step][1]])
        start = step * 8
        assert torch.equal(x, torch.arange(start, start + 8).view(2, 4))
        assert torch.equal(y, torch.arange(start + 1, start + 9).view(2, 4))


def test_shard_rollover_and_recycling(tmp_path):
    np.save(tmp_path / "train_000.npy", np.arange(11, dtype=np.uint16))
    np.save(tmp_path / "train_001.npy", np.arange(20, 29, dtype=np.uint16))
    loader = make_loader(tmp_path, max_tokens=24)

    batches = list(loader)

    assert len(loader) == len(batches) == 3
    for (x, y), start in zip(batches, [0, 20, 0]):
        assert torch.equal(x, torch.arange(start, start + 8).view(2, 4))
        assert torch.equal(y, torch.arange(start + 1, start + 9).view(2, 4))


def test_token_budget_stops_after_full_batches(tmp_path):
    np.save(tmp_path / "train_000.npy", np.arange(33, dtype=np.uint16))
    loader = make_loader(tmp_path, max_tokens=19)

    assert len(loader) == 2
    assert len(list(loader)) == 2


def test_iteration_restarts_from_first_batch(tmp_path):
    np.save(tmp_path / "train_000.npy", np.arange(17, dtype=np.uint16))
    loader = make_loader(tmp_path, max_tokens=16)

    first_pass = list(loader)
    second_pass = list(loader)

    for (first_x, first_y), (second_x, second_y) in zip(first_pass, second_pass):
        assert torch.equal(first_x, second_x)
        assert torch.equal(first_y, second_y)


def test_skips_short_shards_and_other_splits(tmp_path):
    np.save(tmp_path / "train_000.npy", np.arange(4, dtype=np.uint16))
    np.save(tmp_path / "train_001.npy", np.arange(10, 19, dtype=np.uint16))
    np.save(tmp_path / "val_000.npy", np.arange(20, 29, dtype=np.uint16))

    x, y = next(iter(make_loader(tmp_path)))

    assert torch.equal(x, torch.arange(10, 18).view(2, 4))
    assert torch.equal(y, torch.arange(11, 19).view(2, 4))


def test_rejects_batch_size_not_divisible_across_ranks(tmp_path):
    with pytest.raises(ValueError, match="must be divisible"):
        make_loader(tmp_path, world_size=3)
