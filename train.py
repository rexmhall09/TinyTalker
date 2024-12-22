import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
from model import GPTLanguageModel, device, n_embd, n_head, n_layer, dropout, block_size
import os
import itertools

# hyperparameters
batch_size = 64 # how many independent sequences will processing in parallel
max_iters = 30000
learning_rate = 3e-4
print(f"Using device: {device}")
eval_iters = 100 #save iters rn
# ------------


UNK_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'

# Read unique characters from file
with open('chars.txt', 'r', encoding='utf-8') as f:
    charstext = f.read()

# Extract all unique characters that occur in this text
chars = sorted(list(set(charstext)))
chars.extend([UNK_TOKEN, EOS_TOKEN])  # Add special tokens
vocab_size = len(chars)

# Create mappings between characters and integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# Get token IDs for special tokens
unk_id = stoi[UNK_TOKEN]
eos_id = stoi[EOS_TOKEN]

# Encoder: splits on EOS_TOKEN, encodes each part, and adds EOS tokens appropriately
def encode(s):
    parts = [ [stoi.get(c, unk_id) for c in part] + [eos_id] for part in s.split(EOS_TOKEN) ]
    return list(itertools.chain.from_iterable(parts))

# Decoder: preserves EOS tokens in the middle, only adds final EOS if present
decode = lambda l: ''.join([itos.get(i, UNK_TOKEN) if i != eos_id 
                           else (EOS_TOKEN if i == l[-1] or len(l[l.index(i)+1:]) > 0 
                                 else '') 
                           for i in l])



# do this to get shakespeare input.txt wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# split for train and test
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
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

# instantiate the model
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
)
#loads model that will be overwritten b4 training
if os.path.exists("model.pth"):
    print("model.pth exists. Loading the model...")
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
else:
    print("model.pth does not exist. Skipping model loading.")
m = model.to(device)

# print parameters
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a torch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in tqdm.tqdm(range(max_iters)):
    #save model and evaluate the loss on train and val set (not doing loss rn)
    if iter % eval_iters == 0 or iter == max_iters - 1:
    #     losses = estimate_loss()
    #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save(model.state_dict(), "checkpoints/model_epoch_" + str(iter) + ".pth")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model's state dictionary
torch.save(model.state_dict(), "model.pth")
print("Model saved successfully!")
