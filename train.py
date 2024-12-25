import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
from model import GPTLanguageModel, device, n_embd, n_head, n_layer, dropout, block_size
from tokenizer import Tokenizer
import os
import itertools

# Hyperparameters
batch_size = 64  # How many independent sequences will be processed in parallel
max_iters = 30000
learning_rate = 3e-4
print(f"Using device: {device}")
eval_iters = 100  # Save iters rn
# ------------

# Initialize the tokenizer
tokenizer = Tokenizer()

def encode(s):
    """Encodes a string into a list of token IDs using the tokenizer."""
    return tokenizer.encode(s)

def decode(l):
    """Decodes a list of token IDs back into a string using the tokenizer."""
    return tokenizer.decode(l)

# Download and prepare the data
# Note: Ensure 'input.txt' is present in the project directory
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Split for train and test
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # First 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    """Generates a small batch of data of inputs x and targets y."""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """Estimates the loss on both train and validation sets."""
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

# Print number of parameters
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

# Create a torch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in tqdm.tqdm(range(max_iters)):
    # Save model and evaluate the loss on train and val set
    if iter % eval_iters == 0 or iter == max_iters - 1:
        #losses = estimate_loss()
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{iter}.pth")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model's state dictionary
torch.save(model.state_dict(), "model.pth")
print("Model saved successfully!")
