import timm
import torch
import torch.nn as nn
from config import Config

def model_resnet(pretrained=True, num_classes=10):
    model = timm.create_model('resnet50', pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model