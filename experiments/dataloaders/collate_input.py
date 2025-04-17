import random
import torch
from torch.nn.utils.rnn import pad_sequence

def random_causal_collate(batch, max_length, min_length=8):
    input_seqs = []
    label_seqs = []

    for item in batch:
        input_ids = item["input_ids"]

        # Pick a random crop length
        seq_len = random.randint(min_length, min(len(input_ids) - 1, max_length))

        # Random starting point for the crop
        start_idx = random.randint(0, len(input_ids) - seq_len - 1)
        cropped_input = input_ids[start_idx:start_idx + seq_len]
        cropped_label = input_ids[start_idx + 1:start_idx + seq_len + 1]

        input_seqs.append(torch.tensor(cropped_input, dtype=torch.long))
        label_seqs.append(torch.tensor(cropped_label, dtype=torch.long))

    # Pad all sequences in the batch
    input_batch = pad_sequence(input_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
    label_batch = pad_sequence(label_seqs, batch_first=True, padding_value=-100)  # ignore index

    return input_batch, label_batch