import time
import uuid
from flask import Flask, request, jsonify
import torch

# Import TinyTalker components
from tokenizer import Tokenizer
from model import GPTLanguageModel, device, n_embd, n_head, n_layer, dropout, block_size

app = Flask(__name__)

# Initialize the TinyTalker model and tokenizer at startup
tokenizer = Tokenizer()  # loads chars.txt and prepares vocab
model = GPTLanguageModel(vocab_size=tokenizer.vocab_size, 
                         n_embd=n_embd, n_head=n_head, 
                         n_layer=n_layer, block_size=block_size, 
                         dropout=dropout).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()  # set model to evaluation mode (no gradient)

# Define the /chat/completions route
@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True)  # parse JSON request
    if data is None or "messages" not in data:
        return jsonify({"error": "Invalid request, 'messages' field is required"}), 400

    messages = data["messages"]
    # Combine all messages into one prompt string, separated by the <eos> token
    conversation_text = ""
    for msg in messages:
        content = msg.get("content", "")
        conversation_text += content + tokenizer.eos_token

    # Encode the prompt text to token IDs
    context_tokens = tokenizer.encode(conversation_text)
    # Convert to a PyTorch tensor on the appropriate device
    idx = torch.tensor([context_tokens], dtype=torch.long, device=device)
    # Generation parameters (with some defaults)
    max_new_tokens = data.get("max_tokens", 100)      # number of tokens to generate
    temperature = data.get("temperature", 1.0)        # sampling temperature
    # (You can also parse 'top_p', 'n', etc. if needed; here we use default full sampling)

    # Generate tokens from the model until <eos> or max_new_tokens reached
    output_tokens = []  # store newly generated tokens
    model_input = idx  # starts as the context
    # Use no_grad for efficient inference
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Use last block_size tokens as context (model's attention window)
            model_input_cond = model_input[:, -block_size:]
            # Forward pass: get logits for the next token
            logits, _ = model(model_input_cond)
            logits = logits[:, -1, :]             # logits for the last time step
            if temperature != 1.0:
                # Apply temperature scaling to logits
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # probabilities over vocabulary
            # Sample a token from the probability distribution
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            # If end-of-sequence token generated, stop
            if next_token_id == tokenizer.eos_id:
                finish_reason = "stop"
                break
            # Append token to output and also to the model input for next iteration
            output_tokens.append(next_token_id)
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            model_input = torch.cat((model_input, next_token_tensor), dim=1)
        else:
            # If we didn't break from the loop, we hit max_new_tokens limit
            finish_reason = "length"

    # Decode the output tokens to string (the assistant's reply)
    reply_text = tokenizer.decode(output_tokens)
    # Prepare response in OpenAI chat completion format
    response = {
        "id": "chatcmpl-" + uuid.uuid4().hex,            # unique ID for the completion
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "TinyTalker",
        "choices": [
            {
                "index": 0,
                "message": { "role": "assistant", "content": reply_text },
                "finish_reason": finish_reason
            }
        ]
    }
    return jsonify(response)

if __name__ == "__main__":
    # Run the Flask development server on localhost:1234
    app.run(host="127.0.0.1", port=1234)
