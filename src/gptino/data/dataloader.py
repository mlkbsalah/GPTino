import os

from .utils import load_tokens


class DataLoaderLite:
    def __init__(
        self, data_dir, split, B, T, process_rank, world_size, master_process=False
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.world_size = world_size
        self.master_process = master_process
        self.data_dir = data_dir
        assert split in {"train", "val"}

        shards = os.listdir(data_dir)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_dir, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split} in {data_dir}"
        if master_process:
            print(f"Found {len(shards)} {split} shards.")

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.world_size

        if self.current_position + (B * T * self.world_size + 1) > len(self.tokens):
            self.current_shard += 1
            self.current_position = self.process_rank * self.B * self.T
            if self.current_shard >= len(self.shards):
                self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
        return x, y
