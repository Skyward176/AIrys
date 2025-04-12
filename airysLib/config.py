# For standardizing how i handle configurations of model parameters
import torch


class ConfigLlama():
    def __init__(self,
                vocab_size=128_256,
                context_length=131_072,
                emb_dim = 2048,
                n_heads=32,
                n_layers=16,
                hidden_dim = 8192,
                n_kv_groups = 8,
                rope_base = 500_000.0,
                batch_size=1, 
                dtype=torch.float32,
                rope_freq = {
                        "factor": 32.0,
                        "low_freq_factor": 1.0,
                        "high_freq_factor": 4.0,
                        "original_context_length": 8192
                }
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_kv_groups = n_kv_groups
        self.rope_base = rope_base
        self.batch_size = batch_size
        self.dtype = dtype
        self.rope_freq = rope_freq
class ConfigSU():
    def __init__(self, vocab_size=32000, emb_dim = 1024, n_layers=2, n_heads=8, latent_qkv_dim=128, 
                d_rope=16, context_length=1024, batch_size=1, 
                ffn_dim=384, device_groups=4, hidden_dim = 5120, dtype=torch.float32):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.latent_qkv_dim = latent_qkv_dim
        self.d_rope = d_rope
        self.context_length = context_length
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.dtype = dtype

class ConfigDeep():
    def __init__(self, vocab_size=32000, d_in = 5120, d_out=5120, n_layers=2, n_heads=8, latent_qkv_dim=128, 
                d_rope=16, n_experts=32, n_shared=2, top_k=2, seq_len=256, batch_size=1, 
                ffn_dim=384, device_groups=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_in = d_in
        self.d_out = d_out
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.latent_quv_dim = latent_qkv_dim
        self.d_rope = d_rope
        self.n_experts = n_experts
        self.n_shared = n_shared
        self.top_k = top_k
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.ffn_dim = ffn_dim
        self.device_groups = device_groups