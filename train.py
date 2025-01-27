import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
from model import GPTLanguageModel, device, n_embd, n_head, n_layer, dropout, block_size
from tokenizer import Tokenizer
import os
import numpy as np

# Hyperparameters
batch_size = 16
max_iters = 30000
learning_rate = 3e-4
print(f"Using device: {device}")
eval_iters = 300
# ------------

# Initialize the tokenizer
tokenizer = Tokenizer()

def encode(s):
    return tokenizer.encode(s)

def decode(l):
    return tokenizer.decode(l)

# Prepare tokenized data in memory-mapped format
tokenized_file = 'tokenized_data.bin'

# Replace the tokenization section with this chunk-based approach
if not os.path.exists(tokenized_file):
    print("Tokenizing input.txt in chunks and saving to tokenized_data.bin...")
    buffer_size = 1024 * 1024  # Process 1MB at a time (adjust as needed)
    encoded_data = []
    
    with open('input.txt', 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(buffer_size)
            if not chunk:
                break
            # Encode the chunk and extend the encoded data
            encoded_chunk = encode(chunk)
            encoded_data.extend(encoded_chunk)
    
    # Save as numpy array
    encoded_np = np.array(encoded_data, dtype=np.int32)
    with open(tokenized_file, 'wb') as f:
        encoded_np.tofile(f)

# Load memory-mapped data
data_np = np.memmap(tokenized_file, dtype=np.int32, mode='r')
data = torch.from_numpy(data_np).long()

# Split into train and validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Instantiate the model
model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
)
# Load existing model if available
if os.path.exists("model.pth"):
    print("model.pth exists. Loading the model...")
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
else:
    print("model.pth does not exist. Skipping model loading.")
m = model.to(device)

print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in tqdm.tqdm(range(max_iters)):
    if iter % eval_iters == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Iter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{iter}.pth")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pth")
print("Model saved successfully!")
