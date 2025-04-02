import torch
import torch.nn as nn

class NormalizationLayer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # tiny bias to avoid zero divisions
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True) # calculate mean of input vector
        var = x.var(dim=-1,keepdim=True, unbiased=False) #calculate variance of input vector
        
        norm_x = (x-mean) / torch.sqrt(var + self.eps) # normalize with this equation x_norm = x-mean/ sqrt(variance + epsilon)
        
        return self.scale * norm_x + self.shift # output of this is a function with trainable slope and intercept. Normalization can therefore be trained and adjusted to problem