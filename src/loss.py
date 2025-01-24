import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(
        q: torch.Tensor,          # (B, D)
        k_pos: torch.Tensor,      # (B, D)
        k_neg: torch.Tensor,      # (B, N, D)
        temperature: float =0.07
) -> float:
    num = 0
    denom = 0
    final = -torch.log(num / denom)
    return final

