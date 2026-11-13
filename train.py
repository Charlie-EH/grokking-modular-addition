"""
Train the Stage 1 MLP on modular addition; save config, metrics and checkpoints.
Reference: grokking-main notebook, cell "3. Training".
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CONFIG
from data import make_data
from model import MLP
from utils import set_seed, make_run_dir, save_config, log_metrics, save_checkpoint
import argparse


def parse_overrides(config):
    """Let any CONFIG setting be overridden from the command line, e.g. --seed 1."""
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    return vars(parser.parse_args())


CONFIG = parse_overrides(CONFIG)


def evaluate(model, data, loss_fn):
    """Return (loss, accuracy) of the model on a whole dataset, as plain numbers."""
    model.eval()
    with torch.no_grad():
        x = data[:,:2]
        y = data[:,2]
        logits = model(x)
        loss = loss_fn(logits,y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
    return loss.item(), accuracy.item() # .item() turns oen number tensor into python number



# 1. Device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

# 2. Seed
set_seed(CONFIG["seed"])
run_dir = make_run_dir(CONFIG)
save_config(CONFIG, run_dir)


# 3. Data
train_data, test_data = make_data(CONFIG["p"],CONFIG["train_frac"])
train_data = train_data.to(device)
test_data = test_data.to(device)


train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=CONFIG["batch_size"], shuffle=False)
# above not included because evaluate() uses whole test set at once

# 4. Model
model = MLP(CONFIG["p"],CONFIG["d_embed"],CONFIG["d_hidden"])
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=CONFIG["lr"],weight_decay=CONFIG["weight_decay"])


# 5. Training loop

n_epochs = CONFIG["num_epochs"] + 1
for epoch in range(n_epochs):

    # Log: measure train and test simultaneoiusly before trianing
    if epoch % CONFIG["log_every"] == 0:
        train_loss,train_acc = evaluate(model,train_data,loss_fn)
        test_loss,test_acc = evaluate(model,test_data,loss_fn)
        row = {"epoch": epoch,
               "train_loss": train_loss, "test_loss": test_loss,
               "train_acc": train_acc, "test_acc": test_acc}
        log_metrics(run_dir,row)

    # Checkpoint
    if epoch % CONFIG["checkpoint_every"] == 0:
        save_checkpoint(model,run_dir,epoch)

    # train in batches
    model.train()
    for batch in train_loader:
        x = batch[:,:2]
        y = batch[:, 2]
        logits = model(x)
        loss = loss_fn(logits,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# 6. Print where the run was saved.
print(f"Run saved to {run_dir}")