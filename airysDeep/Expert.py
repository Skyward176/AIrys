import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module): # an expert is a feedforward network with 2 linear layers and a GELU activation function in between
    def __init__(self,config):
        super().__init__()
        self.w1 = nn.Linear(config.d_in, config.ffn_dim)
        self.w2 = nn.Linear(config.ffn_dim, config.d_out)
    def forward(self,x):
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        return x