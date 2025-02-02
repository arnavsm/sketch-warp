import os
import gc
import time
import json
import warnings
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Dict, Tuple, List
from logging import getLogger, Logger, \
    INFO, StreamHandler, FileHandler, Formatter

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import timm

from tqdm import tqdm
from PIL import Image
import wandb

from config import Config
from utils import *


augmentations = transforms.Compose([
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=(13, 13), sigma=(0.1, 2.0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])