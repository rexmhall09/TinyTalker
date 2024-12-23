# prompt.py
import torch
import torch.nn.functional as F
from model import GPTLanguageModel, device, n_embd, n_head, n_layer, dropout, block_size
from tokenizer import Tokenizer
import sys

# Initialize the tokenizer
tokenizer = Tokenizer()

# Instantiate model and load weights
model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
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
        prompt = input("Prompt: ") + tokenizer.eos_token
        # Encode the prompt
        context_tokens = tokenizer.encode(prompt)
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
            if next_token_id == tokenizer.eos_id:
                break
            next_char = tokenizer.itos.get(next_token_id, tokenizer.unk_token)
            # Print the generated character
            print(next_char, end='', flush=True)

        print()  # For cleaner formatting
    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except EOFError:
        print("\nExiting.")
        break
