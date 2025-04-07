import torch
import torch.nn as nn

# We use GELU to avoid the non smooth behavior of relu. We also make the negative values (slightly) significant.
# Having a smooth graph also makes it easier to train. It only hits a slope of 0 in one spot, which also makes it faster to fall to its minimum

class GELU(nn.Module): # RELU non linear activation funciton
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return 0.5 * x * (1+torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi)) *
            (x + 0.044715 * torch.pow(x,3)) # using the approximation GELU(x) = 0.5 x (1+tanh(sqrt(2/pi) * (x+0.44715*x^3)))
        ))