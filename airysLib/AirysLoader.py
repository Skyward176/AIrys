import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class AirisDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0,len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i + max_length] # grab a chunk
            target_chunk = token_ids[i+1:i+max_length+1] # the target is the next word

            # turn these into fancy torch tensor array thingies
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self): # implement length method
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx] 
def create_dataloader(txt, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = AirisDataset(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    
    return dataloader