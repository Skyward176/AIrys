import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .RotaryEmbedding import RotaryEmbedding
from .RotaryEmbedding import apply_rotary
class MultiLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_head = config.d_out // config.n_heads
        self.split_dim = self.d_head - config.d_rope

        # Projections for qkv matrices
        # this is the cool part!!!

        #down projection
        self.W_dkv = nn.Linear(config.d_out, config.d_kv_comp)# compress kv down to latent representation of dimension d_kv_comp
        self.W_dq = nn.Linear(config.d_out, config.d_kv_comp) # compress query down to latent representation of dimension d_kv_comp

        #up projection
        self.W_uk = nn.Linear(config.d_kv_comp, config.n_heads * self.split_dim) # expand keys 
        self.W_uv = nn.Linear(config.d_kv_comp, config.n_heads * self.d_head) # expand values
        self.W_uq = nn.Linear(config.d_kv_comp, config.n_heads * self.split_dim) # expand queries

        self.W_qr = nn.Linear(config.d_kv_comp, config.n_heads * config.d_rope) # expand queries to include rope embeddings
        self.W_kr = nn.Linear(config.d_in, config.n_heads * config.d_rope) # expand keys to include rope embeddings

        self.rotary = RotaryEmbedding(config.d_rope)
        self.output = nn.Linear(config.n_heads * self.d_head, config.d_out) # combine heads and project back to d_out
        self.config = config

    def forward(self, input, past_kv=None):
        batch_size, seq_len, _ = input.shape

        #key/value down projection
        c_kv = self.W_dkv(input)
        k = self.W_uk(c_kv).view(batch_size, seq_len, self.config.n_heads, self.split_dim)
        v = self.W_uv(c_kv).view(batch_size, seq_len, self.config.n_heads, self.d_head)

        # queryy down projection
        c_q = self.W_dq(input)
        q_base = self.W_uq(c_q).view(batch_size, seq_len, self.config.n_heads, self.split_dim)
        q_rot = self.W_qr(c_q).view(batch_size, seq_len, self.config.n_heads, self.config.d_rope)

        # Rotary embeddings
        rotary_emb = self.rotary(seq_len)
        cos = torch.cos(rotary_emb).view(1,seq_len, 1, -1)
        sin = torch.sin(rotary_emb).view(1,seq_len, 1, -1)
        
        # Apply ratory embeddings to the query and key tensors
        q_rot = apply_rotary(q_rot, cos, sin)
        k_rot = apply_rotary(
            self.W_kr(input).view(batch_size, seq_len, self.config.n_heads, self.config.d_rope),
            cos, sin
        )

        q = torch.cat([q_base, q_rot], dim=-1)
        k = torch.cat([k, k_rot], dim=-1)

        # Compute attention output

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head) 

        attn = F.softmax(scores, dim=-1)

        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)

        out = self.output(out.contiguous().view(batch_size, seq_len, -1)), (c_kv, k_rot)

        return out