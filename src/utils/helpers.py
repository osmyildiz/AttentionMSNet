"""Utility helpers for DGX environment."""
import os
import random
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return dev
    print("WARNING: CUDA not available, using CPU")
    return torch.device("cpu")


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
