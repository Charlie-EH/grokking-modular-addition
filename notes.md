# Project notes

Short notes per stage. These become the skeleton of the writeup.

## Stage 1: MLP warm-up (finished 4 Nov 2026)

**TODO: write my three-line note before the writeup.**

- What I built:
- What surprised me:
- What I'd change / next question:

Facts to remember for the note:
- Setup: grokking-main MLP settings. p = 53, 50% train, embedding 128, hidden 256,
  batch 64, AdamW lr 3e-4, weight decay 1, 20,000 epochs, seed 42.
- Epoch 0 loss 3.98, matching ln 53 = 3.97 (sanity check passed).
- Weight decay 1: train accuracy 100% by ~150 epochs; test accuracy jumped 0 -> 100% at ~600-1,000.
- Weight decay 0 (control): test accuracy stayed at 0 for all 20,000 epochs; test loss climbed to ~100;
  train loss showed repeated spikes (possibly the "slingshot" effect, Thilak et al. 2022).
- Figure: analysis/figures/stage1_grokking_vs_control.png
- Ideas: weight decay as a spring pulling weights to the origin / a Lagrange multiplier on a norm
  constraint. Back-of-envelope: (1 - lr*wd)^(22 steps x 1,000 epochs) is about e^-6.6, roughly the
  grokking timescale.
