import torch
import torch.nn.functional as F
from model import GPTLanguageModel, device, n_embd, n_head, n_layer, dropout, block_size
import sys

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
encode = lambda s: sum([[stoi.get(c, unk_id) for c in part] + [eos_id] 
                       for part in s.split(EOS_TOKEN)], [])

# Decoder: preserves EOS tokens in the middle, only adds final EOS if present
decode = lambda l: ''.join([
    itos.get(i, UNK_TOKEN) if i != eos_id 
    else (EOS_TOKEN if i == l[-1] else '') 
    for i in l
])

# Instantiate model and load weights
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
).to(device)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

while True:
    try:
        # Prompt the user for input
        prompt = input("Prompt: ")+EOS_TOKEN
        # Encode the prompt
        context_tokens = encode(prompt)
        idx = torch.tensor([context_tokens], dtype=torch.long, device=device)
        generated = idx
        max_new_tokens = 10000  # Set a reasonable limit to prevent infinite loops

        print("Output: ", end='', flush=True)
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = generated[:, -block_size:]
            # Get the predictions
            logits, _ = model(idx_cond)
            # Focus on the last time step
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            generated = torch.cat((generated, idx_next), dim=1)  # (B, T+1)
            # Decode the last token
            next_token_id = idx_next[0].item()
            if next_token_id == eos_id:
                break
            next_char = itos.get(next_token_id, UNK_TOKEN)
            # Print the generated character
            print(next_char, end='', flush=True)

        print()  # For cleaner formatting
    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except EOFError:
        print("\nExiting.")
        break
