import torch
import torch.nn as nn
from torch.nn import LayerNorm
from  airySU.SUTransformer import SUTransformer
# My custom model based on what i've leared from other models.
# My first iteration is to basically clone GPT 2 but use Multi head latent attention and rotary embedding

class airySU(nn.Module):
    def __init__(self, config):
        super().__init__()
        # handle input embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim, dtype=config.dtype) # embedding layer for the tokens
        # transformer blocks
        self.trf_blocks = nn.ModuleList(
            [SUTransformer(config) for _ in range(config.n_layers)]
        )

        # normalize output
        self.final_norm = LayerNorm(config.emb_dim, dtype=config.dtype)

        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False, dtype=config.dtype) # run outputs through a final linear layer

    def forward (self, in_idx): # take the x value of the tokens we're working on
        batch_size, seq_len = in_idx.shape
        x = self.tok_emb(in_idx)
        for block in self.trf_blocks:
            x, _ = block(x, batch_size, seq_len)

        x = self.final_norm(x) # normalize

        logits = self.out_head(x) # turn to logits
        
        return logits