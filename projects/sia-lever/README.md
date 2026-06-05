# SIA-Lever

**When should a self-improving agent update its harness (H) vs its weights (W)?**

Hackathon research project. Phenomenon-first.

## The claim (proven once, measured)
On a 2D-rotation + shortcut-channel symmetry toy (an **adversarial shortcut trap**, not a
naturalistic benchmark):
- A model can hit low prediction error by exploiting a leaked shortcut (fake symmetry win).
- **Weight-only (W) updates PRESERVE the cheating** — structural errors stay high.
- **Harness-first then weight (H→W)** — fix the verifier to detect the shortcut, *then* retrain —
  produces real structural improvement (group-axiom errors collapse to ~0).

Money line, measured not asserted: *"W-only preserved the shortcut failure; H-then-W repaired it."*

## Why symmetry
Fake progress is detectable: a cheater predicts well but fails group-structure tests
(composition, equivariance) and succeeds on a broken-symmetry negative control where an
honest learner must fail. That gives ground truth for which lever was actually needed.

## Structure
- `experiments/` — toy + measurement (`data.py`, `model.py`, `verifier.py`, `train.py`, `run_episode.py`)
- `documentation/plan.md` — phased plan (Phase 0 → 4) + exit gates
- `results/summary.md` — measured numbers as they land
- `figures/` — diffs, plots
- `runs/` — per-run artifacts
- `.state/pipeline.json` — phase + gate tracking

## Execution
Local CPU only. Tiny MLP, seconds per train. No SLURM, no GPU.

## Quick start (Phase 0, once implemented)
```bash
python experiments/run_episode.py --stage 1   # prediction-only baseline
```

## Status
Phase 0 (toy + measurement). See `documentation/plan.md` and `.state/pipeline.json`.
