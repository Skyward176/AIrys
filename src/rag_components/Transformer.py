import torch.nn as nn
import torch

class DecoupledRotaryEmbedding(nn.Module):
    def __init__(self, dim, scale=40a ccxxxc`tesw´`):aqw
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim//2, 2).float() / (dim//2))) # function to calculate the rotation in the complex space. This forms a wave which allows decay over large distances but also allows both fine and coarse grained changes for subtle and large "distances".
        self.register_buffer("inv_freq", inv_freq) # store the inv frequency in a buffer instead of passing it everywhere
        self.scale = scale

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq) / self.scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim =-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin):
    """
    Apply rotary embeddings to the first half of x.
    """
    # Split x into two parts: one for rotary embeddings and the other untouched
    x_rot, x_base = x.split(cos.shape[-1], dim=-1)
    # Apply rotary embeddings to the rotary part
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    # Concatenate the rotary-applied and base parts
    return torch.cat([x_rot, x_base], dim=-1)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_in, d_out, n_heads, latent_qkv_dim, d_rope, dtype):
        super().__init__()
        # config info
        self.n_heads = n_heads
        self.d_rope = d_rope

        self.d_head = d_out // n_heads
        self.split_dim = self.d_head - d_rope
        self.dtype = dtype

        # Projections for qkv matrices
        # this is the cool part!!!

        #down projection
        self.W_dkv = nn.Linear(d_out, latent_qkv_dim, dtype=dtype)# compress kv down to latent representation of dimension d_kv_comp
        self.W_dq = nn.Linear(d_out, latent_qkv_dim, dtype=dtype) # compress query down to latent representation of dimension d_kv_comp

        #up projection
        self.W_uk = nn.Linear(latent_qkv_dim, n_heads * self.split_dim, dtype=dtype) # expand keys 
        self.W_uv = nn.Linear(latent_qkv_dim, n_heads * self.d_head, dtype= dtype) # expand values
        self.W_uq = nn.Linear(latent_qkv_dim, n_heads * self.split_dim, dtype=dtype) # expand queries

        self.W_qr = nn.Linear(latent_qkv_dim, n_heads * d_rope, dtype=dtype) # expand queries to include rope embeddings
        self.W_kr = nn.Linear(d_in, n_heads * d_rope, dtype=dtype) # expand keys to include rope embeddings

        self.rotary = RotaryEmbedding(d_rope)
        self.output = nn.Linear(n_heads * self.d_head, d_out, dtype = dtype) # combine heads and project back to d_out

    def forward(self, input,batch_size, num_tokens, past_kv=None):
        #key/value down projection
        c_kv = self.W_dkv(input)
        k = self.W_uk(c_kv).view(batch_size, num_tokens, self.n_heads, self.split_dim)
        v = self.W_uv(c_kv).view(batch_size, num_tokens, self.n_heads, self.d_head)

        # queryy down projection
        c_q = self.W_dq(input)
        q_base = self.W_uq(c_q).view(batch_size, num_tokens, self.n_heads, self.split_dim)
        q_rot = self.W_qr(c_q).view(batch_size, num_tokens, self.n_heads, self.d_rope)

        # Rotary embeddings
        rotary_emb = self.rotary(num_tokens)
        cos = torch.cos(rotary_emb).view(1,num_tokens, 1, -1)
        sin = torch.sin(rotary_emb).view(1,num_tokens, 1, -1)
        
        # Apply ratory embeddings to the query and key tensors
        q_rot = apply_rotary(q_rot, cos, sin)
        k_rot = apply_rotary(
            self.W_kr(input).view(batch_size, num_tokens, self.n_heads, self.d_rope),
            cos, sin
        )
        del cos, sin

        q = torch.cat([q_base, q_rot], dim=-1)
        k = torch.cat([k, k_rot], dim=-1)

        # Compute attention output

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head) 

        attn = F.softmax(scores, dim=-1)

        out = torch.einsum("bhqk,bkhd->bqhd", attn.to(dtype=self.dtype), v)

        out = self.output(out.contiguous().view(batch_size, num_tokens, -1)), (c_kv, k_rot)

        return out
        
        
        
        
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()