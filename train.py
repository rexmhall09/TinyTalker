import torch
from torch.utils.data import IterableDataset
import os

class ChunkedTextDataset(IterableDataset):
    """
    A streaming dataset that reads a text file line by line, tokenizes the text,
    accumulates tokens in a buffer, and yields them in fixed-size chunks (x, y).
    
    We also allow specifying which portion of the file to read (start_percent, end_percent),
    so that we can simulate train/val splits from the same file in separate Dataset objects.
    """
    def __init__(
        self,
        file_path,
        tokenizer,
        chunk_size=1024,
        start_percent=0.0,
        end_percent=1.0,
        encoding='utf-8'
    ):
        """
        Args:
            file_path (str): Path to your large `input.txt`.
            tokenizer (object): Any callable that does `encode(str) -> List[int]`.
            chunk_size (int): Number of tokens per chunk (e.g. 1024).
            start_percent (float): Fraction of the file (by line count) where this dataset should start reading.
            end_percent (float): Fraction of the file (by line count) where this dataset should stop reading.
            encoding (str): File encoding (usually 'utf-8').
        """
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.encoding = encoding
        
        # You could pre-scan the file to know total number of lines.
        # For large files, scanning once is usually acceptable.
        # This allows us to skip the correct portion for train/val.
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist.")
        
        self.total_lines = 0
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            for _ in f:
                self.total_lines += 1
        
        self.start_line = int(self.total_lines * self.start_percent)
        self.end_line   = int(self.total_lines * self.end_percent)
        # Safety check to avoid edge cases
        self.end_line   = max(self.start_line, self.end_line)
        
    def __iter__(self):
        """
        Iterates through the desired subset of lines in the file, tokenizes them, and yields
        (x, y) pairs of length `chunk_size`.
        """
        token_buffer = []
        current_line_idx = 0
        
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            for line in f:
                # Skip lines before `start_line`
                if current_line_idx < self.start_line:
                    current_line_idx += 1
                    continue
                # Break if we've passed `end_line`
                if current_line_idx >= self.end_line:
                    break
                
                current_line_idx += 1
                
                # Tokenize this line
                tokens = self.tokenizer.encode(line)
                
                # Accumulate tokens
                token_buffer.extend(tokens)
                
                # Yield full-size chunks
                while len(token_buffer) >= self.chunk_size + 1:
                    # We want x and y each of size chunk_size, but y is shifted by 1
                    # so chunk_x = token_buffer[:chunk_size]
                    #    chunk_y = token_buffer[1:chunk_size+1]
                    chunk_x = token_buffer[:self.chunk_size]
                    chunk_y = token_buffer[1:self.chunk_size + 1]
                    
                    # Remove those tokens from the buffer
                    token_buffer = token_buffer[self.chunk_size:]
                    
                    # Convert to tensors
                    x_tensor = torch.tensor(chunk_x, dtype=torch.long)
                    y_tensor = torch.tensor(chunk_y, dtype=torch.long)
                    
                    # Yield (x, y) pair
                    yield x_tensor, y_tensor
            
            # Optionally, handle leftover tokens (usually not needed for LM training).
            # If you want to train on partial chunks at the end of the file,
            # uncomment this block:
            #
            # while len(token_buffer) >= self.chunk_size + 1:
            #     chunk_x = token_buffer[:self.chunk_size]
            #     chunk_y = token_buffer[1:self.chunk_size+1]
            #     token_buffer = token_buffer[self.chunk_size:]
            #     yield torch.tensor(chunk_x, dtype=torch.long), torch.tensor(chunk_y, dtype=torch.long)



import torch
import torch.nn as nn
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model import GPTLanguageModel, device, n_embd, n_head, n_layer, dropout, block_size
from tokenizer import Tokenizer  # same custom Tokenizer you had
import os
import numpy as np

# Hyperparameters
batch_size = 16
max_iters = 100000
learning_rate = 3e-4
eval_iters = 500

print(f"Using device: {device}")

# ------------------------------------------------------------------------------
# 1) Create an instance of our tokenizer
# ------------------------------------------------------------------------------
tokenizer = Tokenizer()

# ------------------------------------------------------------------------------
# 2) Create our two streaming datasets: train and val
# ------------------------------------------------------------------------------
from torch.utils.data import DataLoader

# Import the dataset we just defined above
# from your_datasets_module import ChunkedTextDataset  # if saved in a separate file

train_dataset = ChunkedTextDataset(
    file_path='input.txt',
    tokenizer=tokenizer, 
    chunk_size=block_size, 
    start_percent=0.0,   # start at 0% of lines
    end_percent=0.9      # go up to 90% of lines
)

val_dataset = ChunkedTextDataset(
    file_path='input.txt',
    tokenizer=tokenizer, 
    chunk_size=block_size, 
    start_percent=0.9,   # start at 90%
    end_percent=1.0      # up to 100% of lines
)

# ------------------------------------------------------------------------------
# 3) Create DataLoaders for training and validation
# ------------------------------------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

# ------------------------------------------------------------------------------
# 4) Model Initialization
# ------------------------------------------------------------------------------
model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
).to(device)

# (Optional) Load existing model weights
if os.path.exists("model.pth"):
    print("model.pth exists. Loading the model...")
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
else:
    print("model.pth does not exist. Skipping model loading.")

print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ------------------------------------------------------------------------------
# 5) Helper to estimate train/val loss
# ------------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        # We'll just take `eval_iters` mini-batches from the loader
        # to estimate the average loss.
        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses)/len(losses) if len(losses) > 0 else float('inf')
    model.train()
    return out

# ------------------------------------------------------------------------------
# 6) Training Loop
# ------------------------------------------------------------------------------
for iteration in tqdm.tqdm(range(max_iters)):
    # 6a) Periodically evaluate on train and val
    if iteration % eval_iters == 0 or iteration == max_iters - 1:
        losses = estimate_loss()
        print(f"Iter {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{iteration}.pth")

    # 6b) Get the next batch from the train_loader
    try:
        x_batch, y_batch = next(train_iter)
    except NameError:
        # If we haven't created an iterator yet, create one
        train_iter = iter(train_loader)
        x_batch, y_batch = next(train_iter)
    except StopIteration:
        # If we exhausted the train_loader, recreate the iterator
        train_iter = iter(train_loader)
        x_batch, y_batch = next(train_iter)
    
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    
    # 6c) Forward pass
    logits, loss = model(x_batch, y_batch)
    
    # 6d) Backprop + optimize
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final save
torch.save(model.state_dict(), "model.pth")
print("Model saved successfully!")
