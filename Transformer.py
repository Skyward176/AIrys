import torch.nn as nn
import torch
from MultiHeadAttention import MultiHeadAttention
from FeedForwardLayer import FeedForwardLayer
from NormalizationLayer import NormalizationLayer
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate_attn"],
            qkv_bias = cfg["qkv_bias"]
        ) # attention layer

        self.ff = FeedForwardLayer(cfg) # forward feed layer
        
        self.norm1 = NormalizationLayer(cfg["emb_dim"]) # normalization layers
        self.norm2 = NormalizationLayer(cfg["emb_dim"]) 

        self.drop_shortcut = nn.Dropout(cfg["drop_rate_shortcut"]) # shortcut dropout

    def forward(self,x):
        shortcut = x # attention block shortcut

        x = self.norm1(x) # normalize input
        x = self.att(x) # run attention
        x = self.drop_shortcut(x) 
        
        x = x + shortcut # add in the shortcut value

        return x