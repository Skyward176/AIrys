import torch.nn as nn
from airysDeep.MultiLatentAttention import MultiLatentAttention
import torch
from torch.nn import LayerNorm
from .SUFeedForward import SUFeedForward
from torch.utils.checkpoint import checkpoint
class SUTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config.emb_dim, dtype=config.dtype) # normalization layers
        self.norm2 = LayerNorm(config.emb_dim, dtype=config.dtype) # normalization layers

        self.att = MultiLatentAttention(
            d_in = config.emb_dim,
            d_out = config.emb_dim,
            n_heads = config.n_heads,
            latent_qkv_dim = config.latent_qkv_dim,
            d_rope = config.d_rope,
            dtype = config.dtype,
        ) # attention layer

        self.ff = SUFeedForward(config) # forward feed layer


    def forward(self,x,batch_size,seq_len, past_kv=None):
        shortcut = x
        # x = self.norm1(x)
        # x, new_kv = self.att(x, past_kv)
        x, new_kv = checkpoint(self.att, self.norm1(x), batch_size, seq_len, past_kv, use_reentrant=False) # run attention
        
        x = x + shortcut # add in the shortcut value

        shortcut = x # feed forward shortcut
        # x = self.norm2(x)
        # x = self.ff(x)
        x = checkpoint(self.ff, self.norm2(x),use_reentrant=False)
        x = x + shortcut

        return x, new_kv