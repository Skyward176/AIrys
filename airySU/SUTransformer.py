import torch.nn as nn
from airysDeep.MultiLatentAttention import MultiLatentAttention
import torch
from torch.nn import LayerNorm
from airysGPT.FeedForwardLayer import FeedForwardLayer
from torch.utils.checkpoint import checkpoint
class SUTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config.emb_dim) # normalization layers
        self.norm2 = LayerNorm(config.emb_dim) # normalization layers

        self.att = MultiLatentAttention(
            d_in = config.emb_dim,
            d_out = config.emb_dim,
            n_heads = config.n_heads,
            context_length = config.context_length,
            latent_qkv_dim = config.latent_qkv_dim,
            d_rope = config.d_rope,
            batch_size = config.batch_size,
        ) # attention layer

        self.ff = FeedForwardLayer({"emb_dim": config.emb_dim}) # forward feed layer
        
        self.norm1 = LayerNorm(config.emb_dim) # normalization layers
        self.norm2 = LayerNorm(config.emb_dim) 

        self.drop_shortcut = nn.Dropout(config.drop_rate) # shortcut dropout

    def forward(self,x, past_kv=None):
        shortcut = x
        x, new_kv = checkpoint(self.att, self.norm1(x), past_kv) # run attention
        x = self.drop_shortcut(x) 
        
        x = x + shortcut # add in the shortcut value

        shortcut = x # feed forward shortcut
        x = checkpoint(self.ff, self.norm2(x))
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x, new_kv