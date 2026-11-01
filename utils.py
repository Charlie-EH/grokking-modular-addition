"""
Helper functions used by train.py: seeding, run folders, logging, checkpoints.
"""
import random

import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path


def set_seed(seed):
    """Make every source of randomness start from the same point."""
    random.seed(seed)        # Python's built-in random numbers
    np.random.seed(seed)     # NumPy's random numbers
    torch.manual_seed(seed)  # PyTorch's random numbers (CPU and GPU)

def make_run_dir(config):
    """Create a new, unique folder for this run and return its path."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{stamp}_{config['run_name']}_seed{config['seed']}"
    run_dir = Path(config["runs_dir"])/name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir/"checkpoints").mkdir()
    return run_dir

def save_config(config, run_dir):
    """Save the run's settings as config.json inside its folder."""
    with open(run_dir/"config.json","w") as f:
        json.dump(config,f,indent=2)