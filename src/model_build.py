import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from utils import *

# simplified model architecture
class CorrespondenceModel(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.encoder = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.encoder.fc = nn.Identity()  # Remove final classification layer
        
        self.stn = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 2, 3, padding=1)  
        )

    def forward(self, photo, sketch):
        photo_feat = self.encoder(photo)
        sketch_feat = self.encoder(sketch)
        
        flow = self.stn(photo_feat)
        
        grid = flow.permute(0, 2, 3, 1)  # Reshape for grid_sample
        warped_photo = F.grid_sample(photo, grid, align_corners=True)
        return warped_photo, flow
