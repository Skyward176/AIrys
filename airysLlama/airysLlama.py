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
        x = torch.silu(x_fc1) * x_fc2
        return self.fc3(x)
def precompute_rope_params(head_dim, theta_base=10_000, context_length =4096):

    assert head_dim %2 == 0 

    inv_freq = 1.00 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim //2)].float() / head_dim))