# DG-ANM for EqM — Differential-Geometry-Guided Adversarial Negative Mining

## Research Question
Can EqM be improved by mining adversarial negatives in geometrically meaningful off-manifold directions (normal space), using trajectory failure as the hardness signal?

## Approach
1. Estimate local tangent/normal decomposition around data points via feature-space PCA
2. Mine adversarial perturbations constrained to normal space that induce weak EqM restoring force
3. Train EqM with an auxiliary loss penalizing these failure cases
4. Use autoresearch loop to iterate rapidly (1-epoch CIFAR-10 pilot runs)

## Project Structure
- `eqm-upstream/` — Upstream EqM code (read-only reference)
- `experiments/` — Training and evaluation scripts
- `configs/` — Experiment configurations
- `slurm/` — SLURM submission scripts and logs
- `runs/` — Individual run outputs
- `results/` — Aggregated results and checkpoints
- `documentation/` — Implementation tasks, debugging, experiment queue
- `program.md` — Autoresearch governance file
- `results.tsv` — Autoresearch experiment history

## Quick Start

### Local smoke test
```bash
python projects/diff-EqM/experiments/train_dganm.py --config projects/diff-EqM/configs/baseline.json
```

### Autoresearch (autonomous iteration)
```bash
/autoresearch --project diff-EqM
```

## Status
See `documentation/queue.md` for experiments and `.state/pipeline.json` for pipeline state.
