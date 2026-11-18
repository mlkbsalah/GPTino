import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset

enc = tiktoken.get_encoding("gpt2")


def set_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    return device


class HellaSwagDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_parquet(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return render_example(self.df.iloc[index])


def render_example(example):
    label = int(example["label"])
    ctx_tokens = enc.encode(example["ctx"])
    token_list = []
    mask_list = []
    for end in example["endings"]:
        end_tokens = enc.encode(" " + end)
        token_list.append(ctx_tokens + end_tokens)
        mask_list.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(tokens) for tokens in token_list)

    mask = torch.zeros((len(token_list), max_len))
    tokens = torch.zeros((len(token_list), max_len), dtype=torch.long)

    for i, (tok, mk) in enumerate(zip(token_list, mask_list)):
        tokens[i, : len(tok)] = torch.tensor(tok)
        mask[i, : len(mk)] = torch.tensor(mk)

    return tokens, mask, label
