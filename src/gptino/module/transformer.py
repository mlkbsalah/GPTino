import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, idim, odim, hidden_size, dropout=0.2):
        super().__init__()
        layers = []
        prev_size = idim

        for h_size in hidden_size:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h_size
        layers.append(nn.Linear(prev_size, odim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class AttentionHead(nn.Module):
    def __init__(self, head_size, embed_size, context_window, dropout=0.2):
        super().__init__()
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_window, context_window))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, H)
        k = self.key(x)  # (B, T, H)

        wght = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, H) @ (B, H, T) -> (B, T, T)
        wght = wght.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # type: ignore /causal weights (zero weights for the future)
        wght = F.softmax(wght, dim=-1)
        wght = self.dropout(wght)

        v = self.value(x)  # (B, T, H)
        res = wght @ v  # (B, T, H)

        return res


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_size, context_window, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(embed_size // n_heads, embed_size, context_window)
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, embed_size)
        h = self.dropout(self.proj(h))
        return h


class Block(nn.Module):
    def __init__(self, n_heads, embed_size, context_window):
        super().__init__()
        self.sa = MultiHeadAttention(n_heads, embed_size, context_window)
        self.ffd = MLP(embed_size, embed_size, hidden_size=[4 * embed_size])
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, n_blocks, n_heads, embed_size, context_window):
        super().__init__()
        self.context_window = context_window
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(context_window, embed_size)
        self.blocks = nn.Sequential(
            *[Block(n_heads, embed_size, context_window) for _ in range(n_blocks)]
        )
        self.ln = nn.LayerNorm(embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.position_embedding_table(x)
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln(x)
        logits = self.fc(x)

        return logits

    def generate(self, start_token, max_generated_tokens):
        x = start_token
        for t in range(max_generated_tokens):
            x = x[:, -self.context_window :]
            logits = self(x)
            token_probs = F.softmax(logits[:, -1], dim=-1)
            x_nxt = torch.multinomial(token_probs, 1)
            x = torch.cat((x, x_nxt), dim=1)
        return x


if __name__ == "__main__":
    vocab_size = 10
    n_blocks = 1
    n_heads = 1
    embed_size = 5
    context_window = 10

    model = Transformer(vocab_size, n_blocks, n_heads, embed_size, context_window)

    with torch.no_grad():
        sample = torch.randint(vocab_size, (1, context_window))
        print(model(sample).shape)
        print(model.generate(sample[:, -1:], 30)[0].shape)
