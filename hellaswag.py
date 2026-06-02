import torch.nn.functional as F
import torch
import pandas as pd
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


def iterate_dataset():
    df = pd.read_parquet("./hellaswag/validation-00000-of-00001.parquet")
    for _, row in df.iterrows():
        yield row[["activity_label", "ctx", "endings", "label"]]


def render_example(example):  ## Change the return to tensors
    label = int(example["label"])
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

    return tokens, mask, label


def evaluate(model, device):
    model.to(device)

    total_examples = 0
    acc_sum = 0

    for example in tqdm(iterate_dataset()):
        tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens)[0]  # selecting 0 because returns also loss
        logits = logits[:, :-1, :].contiguous()
        tokens = tokens[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), tokens.view(-1), reduction="none"
        )
        loss = loss.view(tokens.shape[0], -1) * mask[:, 1:]
        avg_loss = loss.sum(dim=1) / mask[:, 1:].sum(dim=1)

        prediction = avg_loss.argmin().item()

        total_examples += 1
        acc_sum += int(prediction == label)
        if total_examples % 100 == 0:
            print(
                f"Current Accuracy ({total_examples}): {acc_sum}/{total_examples}={acc_sum / total_examples:.4f}"
            )

        if total_examples < 5:
            print("-" * 20)
            print("Context:", example["ctx"])
            for i, end in enumerate(example["endings"]):
                print(f"loss {avg_loss[i].item():.4f} : {end}")
            print(f"Predicted: {prediction}, Label: {label}")


if __name__ == "__main__":
    from torch.distributed import init_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import os

    ddp = os.environ.get("RANK", None) is not None
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        device = set_device()
        ddp_rank = 0
        ddp_world_size = 1
        master_process = True
    if master_process:
        print(f"Running on {ddp_world_size} processes. DDP: {ddp}")

    model = GPT.from_pretrained("gpt2")
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    model.eval()
    with torch.no_grad():
        evaluate(raw_model, device)
