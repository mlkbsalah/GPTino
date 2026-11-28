from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .decoder_layer import Block


@dataclass
class GPTConfig:
    context_length: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    activation: Literal["gelu", "swiglu"] = "gelu"
    normalization: Literal["rmsnorm", "layernorm"] = "layernorm"
    pos_encoding: Literal["rope", "learned"] = "learned"


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.RMSNorm(config.n_embd) if config.normalization == "rmsnorm" else nn.LayerNorm(config.n_embd),
            }
        )
        if config.pos_encoding == "learned":
            self.transformer["wpe"] = nn.Embedding(config.context_length, config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer["wte"].weight = self.lm_head.weight
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

    def forward(self, idx):
        _, T = idx.size()
        assert T <= self.config.context_length, (
            f"Cannot forward sequence of length {T}, context length is only {self.config.context_length}"
        )

        x = self.transformer["wte"](idx)
        if self.config.pos_encoding == "learned":
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.transformer["wpe"](pos)[None, :, :]  # broadcast to (B, T, C) and add

        for block in self.transformer["h"]:  # type: ignore
            x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        return logits


    def generate(self, start, max_new_tokens, top_k=None, rng=None):
        idx = start if start.dim() == 2 else start.unsqueeze(0)
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.config.context_length :])  # (B, T, C)
            logits = logits[:, -1, :]  # (B, C)
            if top_k is not None:
                top_k_val = torch.topk(logits, top_k, dim=-1)
                k_th = top_k_val.values[:, -1].unsqueeze(-1)
                logits = logits.where(logits >= k_th, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=rng)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
