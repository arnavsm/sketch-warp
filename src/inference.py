import argparse
import os
import numpy as np

import torch
import torch.nn.functional as F

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
parser.add_argument('--csv-path', metavar='DIR',
                    help='root path to csv files')
parser.add_argument('--data-path', metavar='DIR',
                    help='root path to dataset')
parser.add_argument('--output-dir', metavar='DIR', default='./outputs',
                    help='path to save outputs')


# job
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')

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

test_csv = os.path.join(args.csv_path, "test_pairs_ps.csv")

dataset = PhotoSketchDataset(test_csv, args.data_path, mode="test")
dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

print("Dataset loaded.")

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
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        photo, sketch, _, _ = data

        photo = photo.cuda(non_blocking=True)
        sketch = sketch.cuda(non_blocking=True)

        # get feature maps
        _, photo_res = model.encoder_q(photo, cond=0, return_map=True)
        _, sketch_res = model.encoder_q(sketch, cond=1, return_map=True)

        # estimate displacement field
        fwd_flow, bwd_flow = model.forward_stn(photo_res, sketch_res)
        
        warp_photo = model.spatial_transform(photo, bwd_flow)
        warp_sketch = model.spatial_transform(sketch, fwd_flow)

        # prepare for visualization
        mem = {
            "image1": [prep_img_tensor(photo)],
            "image2": [prep_img_tensor(sketch)],
            "warp_image12": [prep_img_tensor(warp_sketch)],
            "warp_image21": [prep_img_tensor(warp_photo)],
            "weight3_1": [None],
            "weight3_2": [None],
            "dist": [None],
            "res2_1": [photo_res[0].mean(1)],
            "res2_2": [sketch_res[0].mean(1)],
            "res3_1": [photo_res[1].mean(1)],
            "res3_2": [sketch_res[1].mean(1)],
        }
        
        fig = gen_graph(mem)
        fig.savefig(os.path.join(args.output_dir, f'output_{i}.png'), bbox_inches='tight')

print(f"Inference complete. Visualizations saved to {args.output_dir}")
