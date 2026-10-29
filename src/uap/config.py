"""
Configuration, determinism and device selection.

`load_config` resolves `config.yaml` and every path inside it relative to the
project root, so scripts behave identically no matter which directory they are
invoked from. Previously every script had to be run from the project root or it
would fail to find `config.yaml`.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# src/uap/config.py -> src/uap -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PATH_KEYS = (
    "celeba_hq_source", "ffhq_source", "lfw_source",
    "train_data_output", "test_data_output",
    "results_dir", "models_dir", "stargan_weights",
)


def load_config(config_path=None):
    """Load config.yaml and make every path absolute.

    Relative paths in the YAML are resolved against the project root, not the
    current working directory.
    """
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with open(path, "r") as fh:
        config = yaml.safe_load(fh)

    for key in _PATH_KEYS:
        value = config.get("paths", {}).get(key)
        if value and not os.path.isabs(value):
            config["paths"][key] = str((PROJECT_ROOT / value).resolve())
    return config


def seed_everything(seed=42):
    """Seed every RNG the pipeline touches and disable cudnn autotuning."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    """Create a directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path
