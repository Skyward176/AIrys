import torch.nn as nn
from airysDeep.DeepTransformer import DeepTransformer
from torch.nn import LayerNorm as NormalizationLayer

## NOT WORKING RIGHT REEEE
class airysDeep(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_in)
        self.blocks = nn.ModuleList([DeepTransformer(config) for _ in range(config.n_layers)])

        self.norm = NormalizationLayer(config.d_out)
        self.output_head = nn.Linear(config.d_out, config.vocab_size)

        # residiual scaling init

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.1/(config.n_layers**0.5))
        for block in self.blocks:
            block.attn.output.weight.data.mul_(0.1)
            block.MoE.shared_experts[0].w2.weight.data.mul_(0.1)
    def forward(self, input_ids):
        x = self.embed(input_ids)
        total_aux_loss = 0.0
        for block in self.blocks:
            x, _ = block(x)
            total_aux_loss += block.MoE.aux_loss
        x = self.norm(x)
        x = self.output_head(x)
        return x, total_aux_loss