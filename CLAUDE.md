# Grokking project: instructions for Claude

Claude Code reads this file at the start of every session. Keep it up to date.

## Who I am
Edward. Physics & Philosophy MSci (Bristol, 2025). Goal: an alignment/interpretability PhD
(2027–28 entry). This project is my portfolio piece: reproduce grokking on modular addition
(Power et al. 2022) and Nanda et al. 2023's mechanistic analysis and progress measures, in my own code.

- Strong: maths, linear algebra, Fourier analysis, physics, ML concepts (finished Géron's
  *Hands-On ML*, PyTorch edition, through the transformers chapter).
- Weak: **zero software-engineering background.** New to the terminal, git, project
  structure, and writing PyTorch from scratch.

## How to teach me
- Explain every command, flag and symbol the first time it appears: what it does and *why*.
  Assume no knowledge of terminal, git or SE terms. Lab/physics analogies help.
- Maths and ML concepts can go at full speed.
- **One small step at a time.** Give a short chunk, let me run it, wait for me to report back.
- When something errors, explain what the message means, not just the fix.
- **I learn best from diagrams/maps.** Draw structure, flows and data shapes wherever possible.
- **Ask before running commands or editing files**, and say what the command will do first.
  I want to see and understand each step, not have it done silently.

## Working agreement (who writes what)
- **I write:** the data split, model, training loop, Fourier analysis, progress measures,
  and the conclusions in the writeup. Never write these for me. Hint first; give the answer
  only if I ask.
- **Claude may write:** config, file I/O, logging, checkpoint save/load helpers, plot styling,
  README formatting, and other repetitive boilerplate. Explain it line by line.
- **Per-step loop:** orientation (concepts + sources) → I write it → I add sanity checks →
  Claude reviews like a code reviewer → I write a 3-line note on what I learned (feeds the writeup).
- Reproduce the paper's setup exactly before changing anything.
- Full staged plan: see `PLAN.md`.

## Project layout
```
config.py    all settings in one dict (CONFIG); no magic numbers elsewhere
data.py      (a, b) -> (a+b) mod p dataset and train/test split
model.py     the network
train.py     training loop: load config, set seed, train, save outputs
utils.py     seeding, run folders, CSV metrics logger, checkpoint save/load
runs/        one folder per run: config.json, metrics.csv, checkpoints/  (git-ignored)
analysis/    notebooks that only READ from runs/, never retrain
```
Rules: relative paths only, fixed seeds, metrics written to file as training goes,
plots made from saved files. Env is `.venv` (activate with `source .venv/bin/activate`).
Commit little and often with messages that say what changed.

## Status
- Stage 0 (skeleton), second half.
- Done: folder, `.venv`, packages + `requirements.txt`, `.gitignore`, first commit,
  `config.py` (Nanda's settings; `num_epochs` 200 for the dummy run).
- Next: Claude writes `utils.py` (in small explained pieces) → I write a dummy `train.py`
  → check "done when": a run creates a new folder with config.json, metrics.csv and a
  checkpoint; two runs with the same seed give identical metrics.csv; nothing is overwritten;
  a notebook plots the CSV and reloads the checkpoint.

## Cheat sheet
I keep a visual cheat sheet (maps, commands, gotchas) as a claude.ai artifact:
https://claude.ai/artifact/6wpqn9VV6RWr84MRp23mTj
Claude Code can't edit it, so when you draw a new diagram or I hit a new gotcha,
remind me to add it there (or keep a copy in `notes/cheatsheet.md`).
