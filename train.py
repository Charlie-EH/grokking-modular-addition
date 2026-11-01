import torch
from torch import nn

from config import CONFIG
from utils import set_seed, make_run_dir, save_config, log_metrics, save_checkpoint

# 1. Set the seed from CONFIG.
seed = set_seed(CONFIG["seed"])

# 2. Make the run folder and save the config into it.
run_dir = make_run_dir(CONFIG)
save_config(CONFIG, run_dir)

# 3. Fake data: 200 random points with 2 features each, target y = 3*x1 - 2*x2.
#    Use the first 150 as train, the last 50 as test.
X = torch.randn(200,2)
y = 3*X[:,:1]-2*X[:,1:]
X_train, y_train = X[:150],y[:150]
X_test, y_test= X[150:],y[150:] # colons split data into test and train
assert y.shape ==(200,1)

# 4. Model, loss function and optimizer (AdamW, with lr and weight_decay from CONFIG).
model = nn.Linear(in_features=2,out_features=1)
mse = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr = CONFIG["lr"],weight_decay=CONFIG["weight_decay"])


# 5. Training loop over CONFIG["num_epochs"] epochs:
#    - the usual training step on the train set
#    - every log_every epochs: compute test loss, then log a row with
#      epoch, train_loss, test_loss
#    - every checkpoint_every epochs: save a checkpoint

n_steps = CONFIG["num_epochs"] + 1
for epoch in range(n_steps):
    predictions = model(X_train)
    loss = mse(predictions,y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % CONFIG["log_every"] == 0: # checks if a given epoch is a log epoch, if yes, records metrics
        with torch.no_grad():
            test_loss = mse(model(X_test),y_test)      # model on X_test, then mse against y_test
        row = {"epoch": epoch, "train_loss": loss.item(), "test_loss": test_loss.item()}
        log_metrics(run_dir,row)           

    if epoch % CONFIG["checkpoint_every"] == 0:
        save_checkpoint(model,run_dir,epoch)    



# 6. Print where the run was saved.

print(f"Run saved to {run_dir}")
