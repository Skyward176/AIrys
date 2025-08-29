import torch
import torch.nn as nn
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale=40):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim//2, 2).float() / (dim//2)))
        self.register_buffer("inv_freq", inv_freq)
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
    return tozZAZqaaarch.cat([x_rot, x_base], dim=-1)