import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset

enc = tiktoken.get_encoding("gpt2")


class PIQADataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_parquet(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return render_example(self.df.iloc[index])


def render_example(example):
    ctx_tokens = enc.encode(f"Question: {example['goal']}\nAnswer:")
    solutions = [enc.encode(" " + example[key]) for key in ("sol1", "sol2")]
    max_len = len(ctx_tokens) + max(map(len, solutions))
    tokens = torch.zeros((2, max_len), dtype=torch.long)
    mask = torch.zeros((2, max_len))

    for i, solution in enumerate(solutions):
        length = len(ctx_tokens) + len(solution)
        tokens[i, :length] = torch.tensor(ctx_tokens + solution)
        mask[i, len(ctx_tokens) : length] = 1

    return tokens, mask, int(example["label"])
