import yaml
import torch
import random
import numpy as np
import os

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def seed_everything(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Returns the main GPU device or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def ensure_dir(path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)