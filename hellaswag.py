import torch.nn.functional as F
import torch
import numpy as np
from datasets import load_dataset
import tiktoken
from train_gpt2 import GPT
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")


def set_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    return device


def render_example(example):  ## Change the return to tensors
    ctx_tokens = enc.encode(example["ctx"])
    token_list = []
    mask_list = []
    for end in example["endings"]:
        end_tokens = enc.encode(end)
        token_list.append(ctx_tokens + end_tokens)
        mask_list.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(tokens) for tokens in token_list)

    mask = torch.zeros((len(token_list), max_len))
    tokens = torch.zeros((len(token_list), max_len), dtype=torch.long)

    for i, (tok, mk) in enumerate(zip(token_list, mask_list)):
        tokens[i, : len(tok)] = torch.tensor(tok)
        mask[i, : len(mk)] = torch.tensor(mk)

    return tokens, mask


def evaluate(model, device):
    model.to(device)
    losses = []

    ds = load_dataset("Rowan/hellaswag", split="test").select_columns(
        ["activity_label", "ctx", "endings"]
    )
    for example in tqdm(ds):
        tokens, mask = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens)[0]  # selecting 0 because returns also loss
        logits = logits[:, :-1, :].contiguous()
        tokens = tokens[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), tokens.view(-1), reduce=False
        )
        loss = loss.view(tokens.shape[0], -1) * mask[:, 1:]
        loss = loss.sum() / mask[:, 1:].sum()

        losses.append(loss.detach())
    return np.mean(losses)


if __name__ == "__main__":
    device = set_device()
    model = GPT.from_pretrained("gpt2")
    model.eval()
    with torch.no_grad():
        avg_loss = evaluate(model, device)
    print(f"Average loss: {avg_loss.item():.4f}")
