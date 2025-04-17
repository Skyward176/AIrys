import torch
import torch.nn as nn
import torch.nn.functional as F
from .Expert import Expert
class MixOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.shared_experts = nn.ModuleList([Expert(config) for _ in range(config.n_shared)])
        self.routed_experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        self.gate = nn.Linear(config.d_out, config.n_experts)

        self.aux_loss = 0.0
        self.config = config

    def forward(self,x):
        # run the shared experts on all tokens
        shared_out = sum(expert(x) for expert in self.shared_experts)

        # run the routed experts on the top k tokens
        routed_logits = self.gate(x)
        probs = F.softmax(routed_logits, dim=-1)
        topk_probs, topk_indices = probs.topk(self.config.top_k, dim=-1)

        # balance loss among experts
        expert_counts = torch.zeros(self.config.n_experts, device=x.device)
        expert_counts.scatter_add_(0, topk_indices.view(-1), 
                                    torch.ones_like(topk_indices.view(-1), 
                                    dtype=torch.float))

        self.aux_loss += expert_counts.float().var() * 0.003

        # have top k experts process tokens, and combine outputs based on gate scores
        routed_out = torch.zeros_like(x)
        for i in range(self.config.top_k):
            expert_mask = topk_indices[...,i]
            expert_contrib = torch.zeros_like(x)
            
            for expert_idx in range(self.config.n_experts):
                mask = (expert_mask == expert_idx)
                if mask.any():
                    expert_out = self.routed_experts[expert_idx](x[mask])
                    expert_contrib[mask] = expert_out * topk_probs[...,i][mask].unsqueeze(-1)
            routed_out += expert_contrib
        return shared_out + routed_out