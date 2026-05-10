import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters — must match the trained model
BATCH_SIZE = 64
BLOCK_SIZE = 256
N_EMBED = 384
N_HEADS = 6
N_LAYER = 6
DROPOUT = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to {device}")

# vocabulary — must be rebuilt the same way as during training
with open("tinyshakespeare.txt", "r") as f:
    text = f.read()

unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)
stoi = {ch: i for i, ch in enumerate(unique_chars)}
itos = {i: ch for i, ch in enumerate(unique_chars)}


def encode(string):
    return [stoi[c] for c in string]


def decode(index):
    return "".join([itos[i] for i in index])


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        wei = q @ k.transpose(-2, -1)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], -1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, N_HEADS) for _ in range(N_LAYER)])
        self.lnf = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, idx, target=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_embed
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(-1, C)
            target = target.view(-1)
            loss = self.loss_fn(logits, target)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# load and generate
model = BigramLanguageModel()
model.load_state_dict(torch.load("model.pt", map_location=device))
model = model.to(device)
model.eval()

with torch.no_grad():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    completion = decode(model.generate(context, max_new_tokens=10000)[0].tolist())

# save and print
with open("generated.txt", "w") as f:
    f.write(completion)

print(completion)
