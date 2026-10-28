"""
All the settings for a run, in one place.
Change numbers here, never inside the other files.
"""

CONFIG = {
# --- Identity of the run ---
"run_name": "dummy",      # short label; becomes part of the run folder's name
"seed": 0,                # fixes all randomness, so a run can be repeated exactly

# --- Data ---
"p": 113,                 # the modulus in (a + b) mod p
"train_frac": 0.3,        # fraction of all p*p pairs used for training

# --- Training ---
"lr": 1e-3,               # learning rate
"weight_decay": 1.0,
"num_epochs": 200,        # tiny for the dummy run; the real Stage 2 run uses 40_000

# --- Saving ---
"log_every": 10,          # write a row to metrics.csv every N epochs
"checkpoint_every": 100,  # save the model's weights every N epochs
"runs_dir": "runs",       # where run folders get created, relative to the project
}