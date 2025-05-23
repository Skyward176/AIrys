import torch
import torch.nn as nn
from importlib.metadata import version

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.fc2 = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.fc3 = nn.Linear(config.hidden_dim, config.emb_dim, dtype=config.dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = torch.nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)
def precompute_rope_params(head_dim, theta_base=10_000, context_length =4096, freq_config=None):

    assert head_dim %2 == 0 

    inv_freq = 1.00 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim //2)].float() / head_dim))

    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    positions = torch.arange(context_length)

    angles = positions[:,None] * inv_freq[None, :]

    angles = torch.cat([angles,angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)

    x_rotated = (x*cos) + (rotated*sin)

    return x_rotated.to(dtype=x.dtype)

class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 num_heads,
                 num_kv_groups,
                 rope_base = 10_000,
                 rope_config=None,
                 dtype=None
        ):

        super().__init__()
        
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num heads must be divisible by num_kv_groups"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads
        
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias =False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias =False, dtype=dtype)

        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads//num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias =False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype) # Combines head outputs

        mask, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)

        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) # This makes the shape (b, num_tokens,num_heads d_out)

        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Split our matrices by the num_heads dimension
        
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        
        keys = keys.repeat_interleave(self.group_size, dim = 1)
        values = values.repeat_interleave(self.group_size, dim = 1)

        attn_scores = queries @ keys.transpose(2,3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores/ keys.shape[-1]**0.5, dim = -1)# matrix form of scaled dot product attention

        context_vec = (attn_weights@values).transpose(1,2)

        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            num_heads=config.n_heads,
            num_kv_groups=config.n_kv_groups,
            rope_base=config.rope_base,
            rope_config=config.rope_freq,
            dtype=config.dtype
        )
        self.ff = FeedForward(config)

        self.norm1 = torch.nn.RMSNorm(config.emb_dim, eps=1e-5)
        self.norm2 = torch.nn.RMSNorm(config.emb_dim, eps=1e-5)
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x.to(torch.bfloat16))
        
        x = x+shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x+shortcut

        return x
class airysLlama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim, dtype = config.dtype)
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = torch.nn.RMSNorm(config.emb_dim, eps=1e-5)

        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False, dtype = config.dtype)

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        
        x = tok_embeds
        
        x =self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits