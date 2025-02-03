import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass