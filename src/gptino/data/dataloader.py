import os

from .utils import load_tokens
import numpy as np
from gptino.training.trainer import Budget



class DataLoaderLite:
    def __init__(self, data_dir, split, max_tokens, batch_size_tokens, context_length, process_rank, world_size, master_process=False):
        self.context_length = context_length
        self.max_tokens = max_tokens
        self.process_rank = process_rank
        self.world_size = world_size
        self.batch_tokens = batch_size_tokens
        self.master_process = master_process
        self.data_dir = data_dir
        assert split in {"train", "val"}


        self.current_shard = 0
        self.split = split
        self.shards = self._find_shards(data_dir, split)
        

    def _find_shards(self, data_dir, split):
        """
        Find the shards for the given split and return the list of shards (paths) necessary to cover max_tokens
        - if shards are not enough the function returns all the shards available
        """
        shards = sorted(s for s in os.listdir(data_dir) if split in s)
        shards = [os.path.join(data_dir, s) for s in shards]
        assert len(shards) > 0, f"No shards found in {data_dir} for split {split}"

        shard_lenghts = [len(np.load(s, mmap_mode="r")) for s in shards]
        step_tokens = self.batch_tokens * self.world_size
        
        effective_steps_per_shard = [((l-1) // step_tokens) for l in shard_lenghts]
        effective_tokens_per_shard = [steps * step_tokens for steps in effective_steps_per_shard]

        count_at_start = np.cumsum(effective_tokens_per_shard) - effective_tokens_per_shard
        needed_shards = np.where(count_at_start < self.max_tokens)[0][-1]
        return shards[:needed_shards + 1]


    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.process_rank * self.batch_tokens
        self.step_start = 0

    def next_batch(self):
        start = self.step_start + self.process_rank * self.batch_tokens
        buf = self.tokens[start : start + self.batch_tokens + 1]
        x = buf[:-1].view(-1, self.context_length)
        y = buf[1:].view(-1, self.context_length)

        self.step_start += self.batch_tokens * self.world_size
        if self.step_start + self.batch_tokens * self.world_size + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.step_start = 0
        return x, y
    
    def __len__(self):
        needed_steps = self.max_tokens // (self.batch_tokens * self.world_size)
        return needed_steps

    def __iter__(self):
        self.reset()
        for _ in range(len(self)):
            yield self.next_batch()


    