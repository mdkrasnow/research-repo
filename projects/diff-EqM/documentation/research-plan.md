# Research Plan — DG-ANM for EqM

## One-line Summary
Improve Equilibrium Matching by mining adversarial negatives in geometrically meaningful off-manifold directions, using trajectory failure as the hardness signal.

## Stage 1: CIFAR-10 with EqM-S/2
- **Model**: EqM-S/2 (depth=12, hidden=384, patch=4, heads=6, ~6M params)
- **Data**: CIFAR-10 32x32, direct pixel space (no VAE)
- **Pilot**: 1 epoch per experiment (~5 min on A100)
- **Metric**: short_horizon_recovery_distance (feature-space return distance)
- **Mode**: Autoresearch (continuous loop with ratchet)

## Method: DG-ANM
1. Estimate local tangent/normal decomposition via feature-space PCA
2. Mine adversarial negatives constrained to normal space
3. Train with auxiliary loss penalizing weak restoration at negatives

## Key Files
- `experiments/train_dganm.py` — Main training script (MODIFIABLE by autoresearch)
- `experiments/evaluate.py` — Evaluation oracle (IMMUTABLE during autoresearch)
- `program.md` — Autoresearch governance (IMMUTABLE during autoresearch)
- `configs/*.json` — Experiment configurations (MODIFIABLE by autoresearch)

## Ablation Plan (handled by autoresearch)
See `program.md` hypothesis generation strategy for the ordered exploration.
