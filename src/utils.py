import os
import gc
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
from logging import getLogger, INFO, \
    Logger , StreamHandler, FileHandler, Formatter

import numpy as np
import pandas as pd

import torch
from torch import nn

from ptflops import get_model_complexity_info


def seed_everything(seed=42):
    """
    Set a seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # Set the PYTHONHASHSEED environment variable
    np.random.seed(seed)  # Set the seed for NumPy
    torch.manual_seed(seed)  # Set the seed for PyTorch
    torch.cuda.manual_seed(seed)  # Set the seed for CUDA (if using GPU)
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic


def setup_logger(
        model_name: str, 
        logging_dir: str
) -> Logger:
    """
    Set up a logger for tracking model performance and events.
    """
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
    """
    Save the model's state dictionary to a given directory.
    """
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
    """
    Load the model's state dictionary from a given directory.
    """
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


def save_metrics_report(
        report: Dict, 
        model_name: str, 
        epoch: int, 
        save_dir: str
):
    """
    Save the metrics report as a JSON file for each epoch.
    """
    report_dir = os.path.join(save_dir, model_name)
    os.makedirs(report_dir, exist_ok=True) 

    report_filename = (
        f"metrics_epoch_{epoch+1}.json" 
    )
    report_path = os.path.join(report_dir, report_filename)

    # Save the report as a JSON file
    with open(report_path, "w") as report_file:
        json.dump(report, report_file, indent=4)

    print(
        f"[INFO] Saved metrics report for {model_name}, epoch {epoch+1} at {report_path}"
    )


def clear_model_from_memory(model: nn.Module):
    """
    Clears the specified model from memory to free up resources.

    Args:
        model (torch.nn.Module): The PyTorch model to be cleared from memory.
    """
    del model
    gc.collect()
    torch.cuda.empty_cache()


def model_size_mb(model: nn.Module):
    """
    Calculates the size of the model's parameters in megabytes (MB).

    Args:
        model (nn.Module): The PyTorch model whose parameter size is to be calculated.

    Returns:
        float: The size of the model's parameters in megabytes.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_in_bytes = total_params * 4
    size_in_mb = size_in_bytes / (1024**2)
    return size_in_mb


def count_model_flops_params(
    model: nn.Module, 
    input_size: tuple =(3, 224, 224), 
    print_results: bool =True
):
    """
    Calculates and optionally prints the FLOPs (Floating Point Operations) and number of parameters
    for a given PyTorch model.

    Args:
        model (nn.Module): The target PyTorch model for which to compute the FLOPs and parameters.
        input_size (tuple, optional): The size of the input tensor. Default is (3, 224, 224),
                                       which represents a 3-channel image of size 224x224 pixels.
        print_results (bool, optional): If True, prints the calculated FLOPs and parameters.
                                         Default is True.

    Returns:
        tuple: A tuple containing two elements:
            - gflops (float): The computed FLOPs of the model in GFLOPs (Giga Floating Point Operations).
            - params (float): The number of parameters in the model in millions (M).

    Notes:
        - The function uses the `ptflops` library to calculate the number of MACs (Multiply-Accumulate Operations)
          and parameters.
        - MACs are converted to GFLOPs by dividing by 1e9 (1 billion).
        - If an error occurs during computation, it will be printed, and the function will return (None, None).
    """
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
