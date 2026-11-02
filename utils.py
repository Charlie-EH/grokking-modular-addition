"""
Helper functions used by train.py: seeding, run folders, logging, checkpoints.
"""
import random

import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
import csv


def set_seed(seed):
    """Make every source of randomness start from the same point."""
    random.seed(seed)        # Python's built-in random numbers
    np.random.seed(seed)     # NumPy's random numbers
    torch.manual_seed(seed)  # PyTorch's random numbers (CPU and GPU)
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # make gpu do repeatable maths
        torch.backends.cudnn.benchmark = False # prevent randomization


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

def log_metrics(run_dir,row):
    """Append one row to metrics.csv, writing the column names first if the file is new"""
    path = run_dir/"metrics.csv"
    is_new = not path.exists()
    with open(path,"a",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def save_checkpoint(model,run_dir,epoch):
    """Save the model's weights at this epoch into run's checkpoints folder."""
    path = run_dir/"checkpoints"/f"epoch_{epoch:06d}.pt"
    torch.save(model.state_dict(),path)
    return path

def load_checkpoint(model,path):
    """Load saved weights from a checkpoint file inot a model of the same shape."""
    state = torch.load(path,map_location="cpu")
    model.load_state_dict(state)
    return model