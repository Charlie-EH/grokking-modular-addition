"""
Helper functions used by train.py: seeding, run folders, logging, checkpoints.
"""
import random

import numpy as np
import torch


def set_seed(seed):
    """Make every source of randomness start from the same point."""
    random.seed(seed)        # Python's built-in random numbers
    np.random.seed(seed)     # NumPy's random numbers
    torch.manual_seed(seed)  # PyTorch's random numbers (CPU and GPU)