import os
import gc
import random
from pathlib import Path

import torch
from torch import nn
from ptflops import get_model_complexity_info
import numpy as np


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        print(f"GPU Not Available")


def clear_model_from_memory(model: nn.Module):
    del model
    gc.collect()
    torch.cuda.empty_cache()


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


def calculate_model_metrics(
    model: nn.Module, input_size: tuple = (128, 768), print_results: bool = True
):
    try:
        macs, params = get_model_complexity_info(
            model,
            input_size,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )

        gflops = macs / 1e9
        model_size_mb = (params * 4) / (1024**2)
        model_size_gb = model_size_mb / 1024

        if print_results:
            print(f"Computational complexity: {gflops:.3f} GFLOPs")
            print(f"Number of parameters: {params / 1e6:.3f} M")
            print(f"Model size: {model_size_mb:.2f} MB ({model_size_gb:.2f} GB)")

        return gflops, params, model_size_mb

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None, None
