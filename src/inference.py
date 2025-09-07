import argparse
import os
import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from models import encoder
from training import moco

from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import PhotoSketchDataset
from models.correspondence_model import CorrespondenceModel
from utils.visualization import prep_img_tensor, gen_graph

############################
# Argument parser
parser = argparse.ArgumentParser(description='PyTorch Training')

# data
parser.add_argument('--photo-path', metavar='DIR',
                    help='path to photo')
parser.add_argument('--sketch-path', metavar='DIR',
                    help='path to sketch')
parser.add_argument('--output-dir', metavar='DIR', default='./outputs',
                    help='path to save outputs')

# model arch
parser.add_argument('--arch', metavar='ARCH', default='resnet18',
                    choices=['resnet18', 'resnet50', 'resnet101'],
                    help='model architecture')
parser.add_argument('--layer', default=[2, 3], nargs='*', type=int,
                    help='resnet blocks used for similarity measurement')
parser.add_argument('--no-cbn', action='store_false', dest='cbn',
                    help='not use conditional batchnorm')

# checkpoint
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint; resume the entire model')

args = parser.parse_args()

############################
# Initialization
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load images
photo = Image.open(args.photo_path).convert('RGB')
sketch = Image.open(args.sketch_path).convert('RGB')

# Apply transformations
photo = transform(photo).unsqueeze(0).cuda()
sketch = transform(sketch).unsqueeze(0).cuda()

print("Images loaded and transformed.")

# import the original or the conditional BN version of ResNet
if args.cbn:
    resnet = encoder
else:
    raise NotImplementedError("Only ResNet with CBN is supported.")

model = CorrespondenceModel(moco.MoCo, resnet.__dict__[args.arch], dim=128, K=8192, corr_layer=args.layer).cuda()

checkpoint = torch.load(args.checkpoint)
state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):
    if "module." in k:
        state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]

msg = model.load_state_dict(state_dict, strict=False)
assert len(msg.missing_keys) == 0 and len(msg.unexpected_keys) == 0
model = model.cuda().eval()

print("Model loaded.")

############################
# Computation

image_size = 256
with torch.no_grad():
    # get feature maps
    _, photo_res = model.encoder_q(photo, cond=0, return_map=True)
    _, sketch_res = model.encoder_q(sketch, cond=1, return_map=True)

    # estimate displacement field
    fwd_flow, bwd_flow = model.forward_stn(photo_res, sketch_res)
    
    warp_photo = model.spatial_transform(photo, bwd_flow)
    
    # Save the warped photo
    from torchvision.utils import save_image
    output_path = os.path.join(args.output_dir, 'generated_sketch.png')
    save_image(warp_photo, output_path)

print(f"Inference complete. Generated sketch saved to {output_path}")
