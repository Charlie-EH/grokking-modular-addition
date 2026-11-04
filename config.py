"""
All the settings for a run, in one place.
Change numbers here, never inside the other files.

Stage 1 uses the grokking-main MLP settings (a known-good reference).
For Stage 2, Nanda's transformer uses: p=113, train_frac=0.3, full batch,
lr=1e-3, weight_decay=1.0, num_epochs=40_000.
"""

CONFIG = {
    # --- Identity of the run ---
    "run_name": "mlp_wd0",
    "seed": 42,               # same seed as grokking-main

    # --- Data ---
    "p": 53,                  # the modulus in (a + b) mod p
    "train_frac": 0.5,        # fraction of all p*p pairs used for training

    # --- Model ---
    "d_embed": 128,           # length of each number's embedding vector
    "d_hidden": 256,          # neurons in the hidden ReLU layer

    # --- Training ---
    "batch_size": 64,         # pairs per gradient step
    "lr": 3e-4,               # learning rate
    "weight_decay": 0.0,
    "num_epochs": 20_000,

    # --- Saving ---
    "log_every": 10,          # write a row to metrics.csv every N epochs
    "checkpoint_every": 500,  # save the model's weights every N epochs
    "runs_dir": "runs",
}