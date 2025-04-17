import torch

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """
    Generate tokens using the model with optional top-k sampling and temperature scaling.
    """
    for _ in range(max_new_tokens):
        # Truncate the input sequence to the last `context_size` tokens
        idx_cond = idx[:, -context_size:]  # Ensure input length does not exceed `context_size`

        with torch.no_grad():
            logits = model(idx_cond)  # Forward pass through the model
        logits = logits[:, -1, :]  # Focus on the logits for the last token

        # Apply top-k sampling if specified
        if top_k is not None:
            # Keep only the top-k logits
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # Apply temperature scaling if specified
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample from the distribution
        else:
            # Greedy sampling (argmax)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Stop generating if the end-of-sequence token is encountered
        #if eos_id is not None and (idx_next == eos_id):
        #    break

        # Append the new token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx