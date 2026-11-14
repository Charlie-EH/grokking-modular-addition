"""
All the settings for a run, in one place.
Change numbers here, never inside the other files.

Stage 1 uses the grokking-main MLP settings (a known-good reference).
For Stage 2, Nanda's transformer uses: p=113, train_frac=0.3, full batch,
lr=1e-3, weight_decay=1.0, num_epochs=40_000.
"""

PRESETS = {
    "mlp": {
        # --- Identity of the run ---
        "model": "mlp",
        "run_name": "mlp",
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
        "weight_decay": 1.0,
        "beta1": 0.9,             # AdamW default (Stage 1 never changed it)
        "beta2": 0.999,           # AdamW default (Stage 1 never changed it)
        "num_epochs": 20_000,

        # --- Saving ---
        "log_every": 10,          # write a row to metrics.csv every N epochs
        "checkpoint_every": 500,  # save the model's weights every N epochs
        "runs_dir": "runs",
    },

    "transformer": {
        # --- Identity of the run ---
        "model": "transformer",
        "run_name": "tf",
        "seed": 42,               # our choice, same as the MLP runs

        # --- Data ---
        "p": 113,                 # Nanda: modulus 113
        "train_frac": 0.3,        # Nanda: 30% of all pairs used for training

        # --- Model (Nanda: one-layer transformer) ---
        "d_model": 128,           # Nanda: residual stream width 128
        "n_heads": 4,             # Nanda: 4 attention heads
        "d_head": 32,             # Nanda: 32 per head (4 * 32 = 128)
        "d_mlp": 512,             # Nanda: MLP hidden width 512 (4 * d_model)
        "n_ctx": 3,               # Nanda: input is 3 tokens: "a b ="

        # --- Training ---
        "batch_size": None,       # None = full batch (Nanda trains full batch)
        "lr": 1e-3,               # Nanda: AdamW, lr 1e-3
        "weight_decay": 1.0,      # Nanda: weight decay 1
        "beta1": 0.9,             # Nanda: betas (0.9, 0.98)
        "beta2": 0.98,            # Nanda: betas (0.9, 0.98)
        "warmup_steps": 10,       # Nanda: linear lr warmup over 10 steps
        "num_epochs": 40_000,     # Nanda: 40k full-batch epochs

        # --- Saving ---
        "log_every": 10,          # our choice, same as the MLP runs
        "checkpoint_every": 100,  # Nanda saves a checkpoint every 100 epochs
        "runs_dir": "runs",       # same folder as the MLP runs
    },
}

# Kept so existing code doing `from config import CONFIG` still works.
CONFIG = PRESETS["mlp"]
