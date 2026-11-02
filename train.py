"""
Train the Stage 1 MLP on modular addition; save config, metrics and checkpoints.
Reference: grokking-main notebook, cell "3. Training".
"""
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import CONFIG
from data import make_data
from model import MLP
from utils import set_seed, make_run_dir, save_config, log_metrics, save_checkpoint


def evaluate(model, data, loss_fn):
    """Return (loss, accuracy) of the model on a whole dataset, as plain numbers."""
    # Hint: under torch.no_grad():
    #   x = data[:, :2] (the pairs), y = data[:, 2] (the labels)
    #   logits = model(x)
    #   loss = loss_fn(logits, y)
    #   accuracy = (logits.argmax(dim=1) == y).float().mean()
    #     argmax(dim=1) picks the highest-scoring answer for each row;
    #     == y gives True/False per row; .float().mean() turns that into a fraction.
    #   return both with .item()


# 1. Device: "cuda" if a GPU is available, otherwise "cpu". Print which one.

# 2. Seed, run folder, save config (same as the dummy run).

# 3. Data: make_data with p and train_frac from CONFIG, then .to(device) on both.
#    Train DataLoader: DataLoader(train_data, batch_size=..., shuffle=True)
#    (No test DataLoader needed: evaluate() uses the whole test set at once.)

# 4. Model: MLP(...) with sizes from CONFIG, then .to(device).
#    Loss: nn.CrossEntropyLoss().  Optimizer: AdamW, lr and weight_decay from CONFIG.

# 5. Training loop over epochs 0 .. num_epochs (inclusive, as in the dummy run):
#    a) If a log epoch: evaluate on train and on test, then log a row with
#       epoch, train_loss, test_loss, train_acc, test_acc.
#    b) If a checkpoint epoch: save a checkpoint.
#    c) For each batch from the train DataLoader:
#         x = batch[:, :2], y = batch[:, 2]
#         the usual step: forward, loss, backward, step, zero_grad

# 6. Print where the run was saved.
