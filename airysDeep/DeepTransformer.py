import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from airysDeep.MixOfExperts import MixOfExperts
from .MultiLatentAttention import MultiLatentAttention
class DeepTransformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_out)
        self.attn = MultiLatentAttention(config)
        self.norm2 = nn.LayerNorm(config.d_out)
        self.MoE = MixOfExperts(config)
        self.config = config
    def forward(self, x, past_kv=None):
        # Attention
        attn_out, new_kv = checkpoint(self.attn, self.norm1(x), past_kv) # checkpointing to save memory
        #x = self.norm1(x)
        #attn_out, new_kv = self.attn(x, past_kv)
        x = x + attn_out
        # Mix of Experts
        MoE_out = checkpoint(self.MoE, self.norm2(x))
        # x = self.norm2(x)
        # MoE_out = self.MoE(x)
        x = x + MoE_out
        return x, new_kv