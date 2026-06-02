from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import math
import numpy as np
import os

# tiktoken_cache_dir = "/workdir/bensalama/GPTino/tiktoken/"
# os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir



def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    return device


def seed_everything(seed, device):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif device == "mps":
        torch.mps.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor):
        """forward throught the attention block x: (B, T, C)"""
        B, T, C = x.shape
        qkv = self.c_attn(x)  # (B, n_head, T, 3 * n_embd)
        q, k, v = torch.split(
            qkv, split_size_or_sections=self.n_embd, dim=-1
        )  # each of size (B, T, n_embd)

        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(
            1, 2
        )  # (B, n_head, T, hs)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

        # att = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, n_head, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(wei, dim=-1)
        # out = wei @ v  # (B, n_head, T, T) @ (B, n_head, T, hs) -> (B, n_head, T, hs)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)

        return out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Shared weight betwwen token encoder and lm_head
        self.transformer.wte.weight = self.lm_head.weight

        # GPT-2 paper initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "RESIDUAL_SCALE"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb[None, :, :]
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def generate(self, start, max_new_tokens):
        idx = start
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @classmethod
    def from_pretrained(cls, model_type): ## no internet acess on the cluster to download the weights from huggingface, so we have to load them from a local directory where we have saved the weights using the transformers library
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"} 
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained("./GPT2/") 
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        non_decay_param = [p for n, p in param_dict.items() if p.dim() < 1]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": non_decay_param, "weights_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True
        )
        return optimizer


def load_tokens(filename):
    np_tokens = np.load(filename)
    torch_tokens = torch.tensor(np_tokens, dtype=torch.long)
    return torch_tokens

class DataLoaderLite:
    def __init__(self, B, T, process_rank, world_size, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.world_size = world_size
        assert split in {"train", "val"} 

        data_root = "/home/m-ben-salah/repos/GPTino/fineweb-edu-tokenized/"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split} in {data_root}"
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
            self.current_position = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
        return x, y


if __name__ == "__main__":
    import time
    from datetime import datetime

    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    ddp = int(os.environ.get('RANK', -1)) != -1 # if RANK does not exist we get -1 and ddp i false
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = get_device()
    if master_process:
        print(f"Running on {ddp_world_size} processes. DDP: {ddp}")

    seed_everything(1337, device)
    
    enc = tiktoken.get_encoding("gpt2")

    total_batch_size = 524288 # 2**19
    B = 64 # micro batch size per GPU
    T = 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0, "total batch size must be divisible by micro batch size and world size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total batch size: {total_batch_size} | micro batch size: {B} | grad accum steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, world_size=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, world_size=ddp_world_size, split="val")

    torch.set_float32_matmul_precision("high")

    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model


    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19000

    def get_lr(step):
        if step < warmup_steps:
            return max_lr * ((step + 1) / warmup_steps)
        if step > max_steps:
            return min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    optimizer = raw_model.configure_optimizer(
        weight_decay=0.1, learning_rate=6e-4, device=device
    )

    for step in range(max_steps):
        t0 = time.time()

        # validation
        if step % 100 == 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"step {step:5d} | val loss: {val_loss_accum:.6f}")

        # sample from the model
        if step > 0 and step % 100 == 0:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, lenght)
            x = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while x.size(1) < max_length:
                with torch.no_grad():
                    logits, loss = model(x)
                    logits = logits[:, -1, :]  # (B, vocab)
                    probs = F.softmax(logits, dim=-1)  # (B, vocab)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # (B, 50)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B,1)
                    xcol = torch.gather(topk_indices, -1, ix)
                    x = torch.cat((x, xcol), dim=-1)  # (B, length+1)

            for i in range(num_return_sequences):
                tokens = x[i].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} | sample {i}: {decoded} \n {'-'*80}")

        # training step
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_second = (train_loader.B * train_loader.T * train_loader.world_size) / (t1 - t0)
        if master_process:
            print(
                f"step {step:5d} | loss: {loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_second:,.0f}"
            )

    if master_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(raw_model.state_dict(), f"GPT2-model_{timestamp}.pt")

    if ddp:
        dist.barrier()
        destroy_process_group()
