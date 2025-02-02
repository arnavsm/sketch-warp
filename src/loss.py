import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
    
    def forward(self, 
                q: torch.Tensor,          # (B, D)
                k_pos: torch.Tensor,          # (B, D)
                k_neg: torch.Tensor           # (B, N, D)
        ) -> torch.Tensor:
        
        num = torch.exp(torch.sum((q * k_pos), axis=1) / self.temperature)
        q = q.unsqueeze(1)
        denom = num + torch.sum(torch.exp(torch.sum(q * k_neg, dim=2) / self.temperature), dim=1)
        final = -torch.log(num / denom)
        return final
