# Grokking project: learning and progression plan

*Agreed 28 Oct 2026. Work through the stages in order. Only the current stage matters.*

## Working agreement (how Claude should help)
- **Edward writes:** the data split, model, training loop, Fourier analysis, progress measures, and the conclusions in the writeup.
- **AI may write:** plot styling, config/argparse, file I/O, checkpoint save/load helpers, README formatting, and any repetitive code.
- **Per-step loop:**
  1. Orientation: what concepts are needed, and where to find them (Géron chapter, paper section, reference code).
  2. Edward writes it.
  3. Edward adds sanity checks.
  4. Claude reviews the code for correctness and style, as a reviewer would.
  5. Edward writes a 3-line note on what he learned (this feeds the writeup).
- When Edward is stuck: hint first, then the answer if asked. Point out bugs seen in review.
- Priorities: project output matters a lot (PhD season), but learning can't be traded away. Be efficient, but never write the core code for him.
- Reproduce the paper's setup exactly before changing anything. Own code, same experiment.

## Stage 0: Project skeleton
- Repo layout: `data.py`, `model.py`, `train.py`, a config dict, `runs/` (metrics + checkpoints), `analysis/` notebooks.
- Apply the F1 lessons: no hardcoded paths, fixed seeds, metrics logged to a file, plots read from saved files.
- **Done when:** a dummy run writes a metrics file and a checkpoint.

## Stage 1: MLP warm-up (proves the training loop works)
- Task: (a+b) mod p, embeddings → MLP, full-batch AdamW, heavy weight decay.
- Learn: a PyTorch training loop from scratch, train/test split over all pairs, logging.
- Sources: Géron PyTorch-basics and training-DNNs chapters; `grokking-main` as reference.
- Checks: the train and test sets are disjoint and together cover all p² pairs; initial loss ≈ ln p.
- **Done when:** a train/test accuracy curve (log-scale x-axis) shows delayed generalisation, plus a weight decay = 0 control that doesn't grok.

## Stage 2: Transformer reproduction (Nanda mainline)
- Setup: p = 113; input "a b ="; 1 layer; d_model 128; learned positional embeddings; 4 heads of dimension 32; ReLU MLP with 512 units; no LayerNorm; untied embed/unembed; no biases in embed, attention or unembed. Data: 30% of pairs. Training: full batch, AdamW, lr 1e-3, weight decay 1, 40k epochs.
- Write attention by hand rather than using `nn.MultiheadAttention`, because the analysis needs the internals.
- Save checkpoints regularly (e.g. every 100 epochs) for Stage 5.
- Sources: Géron transformer chapter; Nanda §3 and Appendix A; `transformers.py` as reference.
- Checks: parameter count, tensor shapes, initial loss ≈ ln 113 ≈ 4.73.
- **Done when:** Nanda's Figure 2 (loss/accuracy curves) is reproduced with your own model.
- **Stop point:** "Reproduced grokking in a transformer written from scratch."

## Stage 3: Controls and sweeps (folds in Power et al.)
- Weight decay = 0, a training-fraction sweep (Power's headline result), and a few seeds.
- Run sequentially; `vmap` is optional.
- **Done when:** a figure shows when grokking happens and when it doesn't.

## Stage 4: Mechanistic analysis of the final model
- DFT of the embedding matrix → find your key frequencies (they will differ from Nanda's).
- Attention patterns and neuron activations as periodic functions of a and b.
- Logits in the Fourier basis: check for the cos(w(a+b−c)) structure.
- Ablate everything except the key frequencies and show performance holds.
- Sources: Nanda §4 and the appendices; ARENA's grokking/modular-arithmetic exercises; your own Fourier/representation-theory background.
- **Stop point:** a genuine interpretability result on your own model.

## Stage 5: Progress measures over training
- Restricted loss, excluded loss, sum of squared weights, and the Gini coefficient of Fourier norms, computed across your checkpoints.
- Identify the memorisation → circuit formation → cleanup phases and their epoch boundaries.
- **Stop point:** a full reproduction of Nanda et al.

## Stage 6: Writeup and polish (start a draft at Stage 2)
- README with figures; a blog or LessWrong-style post; clear claims and the controls behind them.

## Stage 7: Extension (only if time allows, pick one)
- Progress measures on the Stage 1 MLP: do the same phases appear without attention?
- First-order logic tasks (syntax vs semantics).
