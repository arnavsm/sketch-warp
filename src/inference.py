import torch
import torch.nn as nn
import torch.nn.functional as F
from model_build import CorrespondenceModel
from utils import *


if __name__ == '__main__':
    model = CorrespondenceModel().eval()
    photo = torch.randn(1, 3, 256, 256)  # Example input (normalized)
    sketch = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        warped_photo, flow = model(photo, sketch)
