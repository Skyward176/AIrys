import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


class AirisDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, dtype=torch.float32):
        self.input_ids = []
        self.target_ids = []
        token_ids = []
        for i in txt:
            token_ids.append(tokenizer.encode(i))

        for i in range(0,len(token_ids)-max_length, stride):
            chunk_size = torch.randint(1, max_length-1, (1,)).item()  # Random size between 1 and max_length-1
            input_chunk = token_ids[i:i + chunk_size]  # Grab a random-sized chunk
            target_chunk = token_ids[i+1:i+chunk_size+1]  # The target is the next word

            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __len__(self): # implement length method
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx] 
def create_dataloader(txt, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers = 0, dtype=torch.float32):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = AirisDataset(txt, tokenizer, max_length, stride, dtype=dtype)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader
class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]

        # Shift labels one token to the left
        labels = input_ids[1:].clone()
        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1]

        return input_ids, attention_mask, labels