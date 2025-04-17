import torch.nn as nn
class SUFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.fc2 = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.fc3 = nn.Linear(config.hidden_dim, config.emb_dim, dtype=config.dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)