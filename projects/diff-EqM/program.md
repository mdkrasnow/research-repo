# program.md — Autoresearch Governance for DG-ANM-EqM
#
# This file governs autonomous experiment iteration for the diff-EqM project.
# The agent reads this at the start of each iteration to decide what to try next.
# Human writes this file; agent writes code.
#
# Hypothesis: EqM can be improved by mining adversarial negatives in geometrically
# meaningful off-manifold directions (normal space), using trajectory failure as
# the hardness signal.

## Objective

metric: short_horizon_recovery_distance
direction: minimize
eval_command: "python projects/diff-EqM/experiments/evaluate.py --config projects/diff-EqM/configs/eval.json"
eval_grep: "^short_horizon_recovery_distance:"

## Baseline

baseline: 0.045237
best_so_far: 0.010786
best_commit: 709bb0a

## Constraints

max_runtime_seconds: 600
max_slurm_minutes: 30
files_allowed:
  - projects/diff-EqM/experiments/train_dganm.py
  - projects/diff-EqM/configs/*.json
files_readonly:
  - projects/diff-EqM/program.md
  - projects/diff-EqM/experiments/evaluate.py
  - projects/diff-EqM/eqm-upstream/
partition: gpu_test
pilot_steps: 1
full_steps: 80

## Hypothesis Generation

exploration_dimensions:
  - mining_loss_weights        # lambda_N, lambda_T, lambda_W, lambda_A, lambda_R balance
  - mining_budget              # epsilon (perturbation radius), T (ascent steps)
  - trajectory_rollout         # L (rollout length), step size for short-horizon EqM descent
  - geometry_estimation        # k (neighbors for PCA), r (tangent rank), feature layer choice
  - negative_loss_design       # margin m, rho, tau weights in L_neg
  - training_hyperparameters   # gamma (neg loss weight), learning rate, batch size
  - architecture               # Which EqM-E variant (none/dot/l2), c(t) schedule

strategy: |
  1. First iteration: run baseline EqM-S/2 on CIFAR-10 (1 epoch) with NO mining.
     This establishes the baseline short_horizon_recovery_distance.
  2. Add simplest mining: random normal-space perturbations with L_weak only.
     If this improves recovery, geometry matters. If not, check geometry estimation.
  3. Add adversarial search (PGA on mining objective) one component at a time:
     L_normal → L_weak → L_align → L_traj. Each in isolation first.
  4. Once individual components validated, combine the best subset.
  5. Sweep mining budget (epsilon, T) to find sweet spot.
  6. Sweep trajectory rollout length L.
  7. Sweep negative loss weight gamma.
  8. ONE change per experiment. Never modify multiple dimensions simultaneously.
  9. If 3 consecutive failures on a dimension, move to next dimension.
  10. Read results.tsv to avoid repeating failed approaches.

## Ratchet Rules

keep_threshold: 0.0
revert_on_regression: true
revert_on_crash: true
max_consecutive_failures: 10
simplicity_preference: true
parallel_candidates: 5

## Termination

max_iterations: 100
target_metric: null
max_wall_hours: 24
stop_on_plateau: true
plateau_window: 15

## Execution Mode

mode: slurm

## Notes

# Stage 1: CIFAR-10 with EqM-S/2 (smallest model, ~6M params, single A100)
# Pilot runs: 1 epoch of CIFAR-10 (~5 min on A100)
# Primary metric: short_horizon_recovery_distance (avg feature-space distance
#   after L EqM gradient descent steps from mined negatives back toward anchor)
# Lower = better (EqM field more effectively restores off-manifold points)
#
# The evaluation script (evaluate.py) is IMMUTABLE during autoresearch.
# It loads the trained model, generates mined negatives in normal space,
# runs L steps of EqM GD from each negative, and measures avg return distance.
#
# Feature space for geometry estimation: EqM's own intermediate activations
# (self-supervised, no external feature extractor needed for CIFAR-10).
#
# Key design: train_dganm.py contains ALL modifiable training logic.
# configs/*.json contain all hyperparameters.
# evaluate.py is the immutable evaluation oracle.
