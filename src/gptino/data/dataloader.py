import os

import numpy as np
import torch


def load_tokens(filename):
    np_tokens = np.load(filename)
    torch_tokens = torch.tensor(np_tokens, dtype=torch.long)
    return torch_tokens

class DataLoaderLite:
    def __init__(
        self,
        data_dir,
        split,
        max_tokens,
        batch_tokens,
        context_length,
        process_rank,
        world_size,
        master_process=False,
    ):
        if batch_tokens % (context_length * world_size) != 0:
            raise ValueError(
                f"batch_tokens {batch_tokens} must be divisible by context_length * world_size = {context_length * world_size}"
            )

        self.context_length = context_length
        self.max_tokens = max_tokens
        self.process_rank = process_rank
        self.world_size = world_size
        self.batch_tokens = batch_tokens
        self.rank_tokens = batch_tokens // world_size
        self.master_process = master_process
        self.data_dir = data_dir
        assert split in {"train", "val"}

        self.current_shard = 0
        self.split = split
        self.shards = self._find_shards(data_dir, split)

    def _find_shards(self, data_dir, split):
        shards = sorted(s for s in os.listdir(data_dir) if split in s)
        shards = [os.path.join(data_dir, s) for s in shards]
        assert len(shards) > 0, f"No shards found in {data_dir} for split {split}"

        shard_lenghts = [len(np.load(s, mmap_mode="r")) for s in shards]

        effective_steps_per_shard = [
            ((length - 1) // self.batch_tokens) for length in shard_lenghts
        ]
        effective_tokens_per_shard = [
            steps * self.batch_tokens for steps in effective_steps_per_shard
        ]

        count_at_start = (
            np.cumsum(effective_tokens_per_shard) - effective_tokens_per_shard
        )
        needed_shards = np.where(
            (count_at_start < self.max_tokens)
            & (np.array(effective_tokens_per_shard) > 0)
        )[0]
        return [shards[i] for i in needed_shards]

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.process_rank * self.rank_tokens
        self.step_start = 0

    def next_batch(self):
        start = self.step_start + self.process_rank * self.rank_tokens
        buf = self.tokens[start : start + self.rank_tokens + 1]
        x = buf[:-1].view(-1, self.context_length)
        y = buf[1:].view(-1, self.context_length)

        self.step_start += self.batch_tokens
        if self.step_start + self.batch_tokens + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.step_start = 0
        return x, y

    def __len__(self):
        return self.max_tokens // self.batch_tokens

    def iter_from(self, step):
        if step == len(self):
            return

        shard_lenghts = [len(np.load(s, mmap_mode="r")) for s in self.shards]
        steps_per_shard = [
            ((length - 1) // self.batch_tokens) for length in shard_lenghts
        ]
        cum_steps = np.cumsum(steps_per_shard)

        cycle_step = step % cum_steps[-1]
        self.current_shard = np.searchsorted(cum_steps, cycle_step, side="right")
        self.tokens = load_tokens(self.shards[self.current_shard])

        step_in_shard = (
            cycle_step
            if self.current_shard == 0
            else cycle_step - cum_steps[self.current_shard - 1]
        )
        self.step_start = step_in_shard * self.batch_tokens

        for _ in range(len(self) - step):
            yield self.next_batch()

    def __iter__(self):
        self.reset()
        return self.iter_from(0)
