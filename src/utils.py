import os
import gc
import random
from datetime import datetime
from pathlib import Path
from typing import Dict
from logging import getLogger, INFO, Logger , StreamHandler, FileHandler, Formatter
import torch
from torch import nn
from torchmetrics.functional import accuracy as torch_accuracy
from ptflops import get_model_complexity_info
import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    try:
        torch.cuda.manual_seed(seed)  
        torch.backends.cudnn.deterministic = True
    except Exception as e:
        print(f"GPU Not Available")
        return None, None

def setup_logger(
        model_name: str, 
        logging_dir: str
) -> Logger:
    log_dir = os.path.join(logging_dir, model_name)
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = getLogger(model_name)

    # Only set up the logger if it hasn't been set up before
    if not logger.handlers:
        logger.setLevel(INFO)

        file_handler = FileHandler(log_file)  # Save logs to file
        stream_handler = StreamHandler()  # Output logs to console

        formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def save_model(
        model: nn.Module, 
        target_dir: str, 
        model_name: str
):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(
        parents=True, exist_ok=True
    )  # Create the directory if it doesn't exist

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"

    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(
        obj=model.state_dict(), f=model_save_path
    )  # Save the model's state dictionary


def load_model(
        model: nn.Module, 
        target_dir: str, 
        model_name: str
):
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    target_dir_path = Path(target_dir)

    model_load_path = target_dir_path / model_name
    assert model_load_path.is_file(), f"Model file not found at: {model_load_path}"

    print(f"[INFO] Loading model from: {model_load_path}")
    model.load_state_dict(
        torch.load(model_load_path)
    )  # Load the model's state dictionary
    return model


def clear_model_from_memory(model: nn.Module):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def model_size_mb(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_in_bytes = total_params * 4
    size_in_mb = size_in_bytes / (1024**2)
    return size_in_mb


def count_model_flops_params(
    model: nn.Module, 
    input_size: tuple =(3, 224, 224), 
    print_results: bool =True
):
    try:
        macs, params = get_model_complexity_info(
            model,
            input_size,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )

        gflops = macs / 1e9  # Convert MACs to GFLOPs

        if print_results:
            print(f"Computational complexity: {gflops:.3f} GFLOPs")
            print(f"Number of parameters: {params / 1e6:.3f} M")

        return gflops, params
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        batch_size = target.size(0)
        res = []
        
        for k in topk:
            # Use torchmetrics accuracy function
            # Convert to float32 to ensure compatibility
            acc = torch_accuracy(
                output.float(), 
                target,
                task="multiclass",
                num_classes=output.size(1),
                top_k=k
            )
            # Multiply by 100 to match original function's percentage output
            res.append(acc.mul_(100.0))
        return res


class AverageMeter(object):
    def __init__(self, name, fmt=':f', category="train"):
        self.name = name
        self.category = category
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def log(self, writer, n):
        writer.add_scalar(self.category + "/" + self.name, self.val, n)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'