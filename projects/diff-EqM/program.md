# program.md — Autoresearch Governance for DG-ANM-EqM (ImageNet-100 FID Proxy)
#
# This file governs autonomous experiment iteration for the diff-EqM project.
# The agent reads this at the start of each iteration to decide what to try next.
# Human writes this file; agent writes code.
#
# Hypothesis: DG-ANM (adversarial mining of off-manifold negatives) can be further
# improved beyond the autoresearch-found gamma=4.0, eps=0.3 config. We evaluate on
# a fast 2-epoch IN-100 FID proxy — directly what we care about, unlike the earlier
# CIFAR recovery-distance proxy which may not transfer to ImageNet FID.
#
# Previous CIFAR autoresearch: 0.045237 → 0.003458 (9 rounds) → translated to FID
# 112.58 on full 80ep IN-100. This new loop targets the IN-100 FID directly.

## Objective

metric: imagenet100_fid
direction: minimize
eval_command: "bash projects/diff-EqM/slurm/jobs/autoresearch_in100_pilot.sbatch"  # Submitted via sbatch with CONFIG_PATH env var
eval_grep: "^imagenet100_fid:"

## Baseline

# Baseline FID measured from pilot 6123455 (2ep IN-100, bs=16, 2K samples)
# Current best DG-ANM params: gamma=4.0, epsilon=0.3, mining_steps=3, mine_every=5
baseline: 278.6632
best_so_far: 278.6632
best_commit: 9c291cd

## Constraints

max_runtime_seconds: 5400   # 90 min per pilot (train 2ep IN-100 4-GPU + sample 5K + FID)
max_slurm_minutes: 90
files_allowed:
  - projects/diff-EqM/experiments/train_imagenet.py
  - projects/diff-EqM/configs/autoresearch_in100_*.json
files_readonly:
  - projects/diff-EqM/program.md
  - projects/diff-EqM/slurm/jobs/autoresearch_in100_pilot.sbatch
  - projects/diff-EqM/slurm/jobs/compute_in100_reference_stats.sbatch
  - projects/diff-EqM/eqm-upstream/
  - projects/diff-EqM/results/in100_reference_stats.npz
partition: gpu_test
pilot_epochs: 2
full_epochs: 80   # Full-scale confirmation runs (on winners only)
num_fid_samples_pilot: 2000
num_fid_samples_full: 50000

## Hypothesis Generation

exploration_dimensions:
  - mining_strength       # gamma (weight of negative loss): try 2.0, 3.0, 4.0, 5.0, 8.0
  - mining_budget         # epsilon (perturbation radius): 0.1, 0.2, 0.3, 0.5
  - mining_search         # mining_steps (PGA ascent steps): 1, 3, 5, 7
  - mining_schedule       # mine_every (how often to mine negatives): 1, 3, 5, 10
  - negative_margin       # neg_margin: 1.0, 3.0, 5.0, 10.0
  - learning_rate         # lr: 5e-5, 1e-4, 2e-4
  - batch_size            # global_batch_size: 128, 256, 512 (larger batches may stabilize mining)

strategy: |
  1. Measure baseline FID with current best config (gamma=4.0, eps=0.3, etc).
  2. One-dimension-at-a-time sweeps on the 7 exploration dimensions above.
     Start with mining_strength (gamma) since it had the biggest effect in CIFAR rounds.
  3. Each round: 3 parallel candidates exploring one dimension.
  4. KEEP rule: a candidate is kept if its FID is >= 1.0 FID points lower than baseline
     (this accounts for FID noise on 5K samples, which is typically ±1-2).
  5. After 1-dim sweeps converge, do 2-dim combinations of winning dimensions.
  6. ONE change per experiment. Never modify multiple dimensions simultaneously.
  7. If 3 consecutive failures on a dimension, move to the next.
  8. Read results.tsv to avoid repeating failed approaches.

## Ratchet Rules

keep_threshold: 1.0           # Must improve by at least 1.0 FID (signal > noise)
revert_on_regression: true
revert_on_crash: true
max_consecutive_failures: 10
simplicity_preference: true
parallel_candidates: 3        # 3 × 4 GPUs = 12 GPUs concurrent

## Termination

max_iterations: 30            # ~30h wall-clock at 3 parallel × 1h per candidate
target_metric: null           # No specific target; best-effort improvement
max_wall_hours: 48
stop_on_plateau: true
plateau_window: 10

## Execution Mode

mode: slurm

## Notes

# Proxy metric: IN-100 FID at 2 epochs, 5K samples.
# Absolute values will be HIGH (1-epoch runs on IN-100 previously got ~360 FID;
# 2 epochs should be lower but still far from the 80ep FID of 112). What matters
# is relative ordering and improvement delta.
#
# Data source: ~/in100_subset/ (symlinks to first 100 IN-1K classes alphabetically,
# 129,395 training images). This is a deterministic subset — not the canonical
# Tian et al. IN-100 — but it's self-consistent for the autoresearch loop.
#
# Reference FID stats: 10K real images (100 per class) precomputed in
# results/in100_reference_stats.npz. IMMUTABLE once computed.
#
# Training is 4-GPU DDP (EqM-B/2, bs=256, 2 epochs ≈ ~900 steps ≈ ~7 min).
# Sampling 5K images: ~3 min.
# FID compute: ~1 min.
# Total per pilot: ~15 min wall-clock (plus queue wait).
#
# IMPORTANT: the autoresearch_in100_pilot.sbatch is IMMUTABLE. It:
#   1. Clones repo at candidate's SHA
#   2. Parses candidate's config JSON for training args
#   3. Trains for 2 epochs on IN-100 subset
#   4. Samples 5K images with GD sampler (stepsize=0.003, 250 steps, cfg=1.0)
#   5. Computes FID against precomputed reference stats
#   6. Prints "imagenet100_fid: <value>" to stdout (grepped by autoresearch)
#
# Candidates modify train_imagenet.py (new training/mining logic) and/or their
# autoresearch_in100_*.json config (hyperparameters).
