import torch
import torch.nn as nn
from NormalizationLayer import NormalizationLayer as NormLayer
from Transformer import Transformer
class Airys(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # handle input embedding
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) 
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # transformer blocks
        self.trf_blocks = nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg["n_layers"])]
        )

        # normalize output
        self.final_norm = NormLayer(cfg["emb_dim"])

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) # run outputs through a final linear layer

    def forward (self, in_idx): # take the x value of the tokens we're working on
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # do this to make sure that this runs where ever the data is, CPU or GPU

        x = tok_embeds + pos_embeds # input vector is batch_size * num_tokens * emb_size
                                    # we've added together the input and positional embeddings to give us a context vector
        x = self.drop_emb(x) # embed dropout

        x = self.trf_blocks(x) # run transformer blocks
        
        x = self.final_norm(x) # normalize

        logits = self.out_head(x) # turn to logits
        
        return logits
        
