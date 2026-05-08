import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 10000
LR = 1e-3
EVAL_INTERVAL = 400
EVAL_ITERS = 200
N_EMBED = 32
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu" 
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

# Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, idx, target=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        x = tok_emb + pos_embed # B, T, C
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

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

loss_list = []
for epoch_i in range(MAX_ITERS):
    if epoch_i%EVAL_INTERVAL==0:
        out = estimate_loss()
        print(f"Epoch {epoch_i:>4}/{MAX_ITERS}: Train loss: {out["train"]:.3f} | Test loss: {out["test"]:.3f}")

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1),dtype=torch.long, device=device)
completion = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(completion)