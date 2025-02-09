import os
import gc
import random
from datetime import datetime
from pathlib import Path
from typing import Dict
import logging
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


def setup_logger(
    log_file: str = "training.log", level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger("ModelTraining")
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure the directory exists
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_model(model: nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"

    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def load_model(model: nn.Module, target_dir: str, model_name: str):
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    target_dir_path = Path(target_dir)

    model_load_path = target_dir_path / model_name
    assert model_load_path.is_file(), f"Model file not found at: {model_load_path}"

    print(f"[INFO] Loading model from: {model_load_path}")
    model.load_state_dict(torch.load(model_load_path))
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
    model: nn.Module, input_size: tuple = (3, 224, 224), print_results: bool = True
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
            acc = torch_accuracy(
                output.float(),
                target,
                task="multiclass",
                num_classes=output.size(1),
                top_k=k,
            )
            res.append(acc.mul_(100.0))
        return res
