import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from icecream import ic

# hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
LR = 3e-4
EVAL_INTERVAL = 400
EVAL_ITERS = 200
N_EMBED = 384
N_HEADS = 6
N_LAYER = 6
DROPOUT = 0.2
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device set to {device}")

torch.manual_seed(42)

# data loading
with open("tinyshakespeare.txt", "r") as f:
    text = f.read()

# Maping between characters and intergers
unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)
stoi = {ch: i for i, ch in enumerate(unique_chars)}
itos = {i: ch for i, ch in enumerate(unique_chars)}
def encode(string):
    return [stoi[c] for c in string]
def decode(index):
    return "".join([itos[i] for i in index])

#train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]

# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    start_ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in start_ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in start_ix])
    x, y = x.to(device), y.to(device)
    return x, y

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
        q = self.query(x) # (B, T, C)
        k = self.key(x) # (B, T, C)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
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
            nn.Dropout(DROPOUT)
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

# Bigram model
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

        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        x = tok_emb + pos_embed # B, T, C
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.lm_head(x) # B, T, vocab_size
        
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
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return  idx
    

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(EVAL_ITERS)
        for iter in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[iter] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

loss_list = []
for epoch_i in tqdm(range(MAX_ITERS)):
    if epoch_i%EVAL_INTERVAL==0:
        out = estimate_loss()
        print(f"Epoch {epoch_i:>4}/{MAX_ITERS}: Train loss: {out["train"]:.3f} | Test loss: {out["test"]:.3f}")

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pt")

context = torch.zeros((1,1),dtype=torch.long, device=device)
completion = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(completion)