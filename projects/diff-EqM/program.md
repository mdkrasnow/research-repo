# program.md — Autoresearch Governance for DG-ANM-EqM (Variant Search)
#
# This file governs autonomous experiment iteration for the diff-EqM project.
# The agent reads this at the start of each iteration to decide what to try next.
# Human writes this file; agent writes code.
#
# THESIS: DG-ANM as currently implemented (PGA mining at hardcoded t=1 +
# feature-normal/tangent projector + margin-hinge on ||field||) loses to vanilla
# EqM by ~2x on CIFAR-10 (DG-ANM 150ep 10K-FID ~21.7 vs vanilla ~9.5). A
# diagnostic (experiments/diag_dganm_signal.py) and a recent literature review
# (see documentation/research-plan.md) both identify the *formulation* — not the
# hyperparameters — as broken. This program abandons hyperparameter sweeps and
# instead deploys *code variants of DG-ANM*, each a concrete reformulation
# motivated by a specific paper, and compares them head-to-head on the same
# CIFAR proxy.
#
# NORTH STAR: publish at NeurIPS / ICML / ICLR. See
# documentation/publishability-plan.md. Stage A's original exit gate
# ("proxy config beats vanilla by >=1 FID across 3 seeds") still holds. We
# cannot pass that gate today, so we are redesigning the primitive.

## Objective

metric: cifar10_variant_fid
direction: minimize
eval_command: "sbatch projects/diff-EqM/slurm/jobs/variant_pilot.sbatch"
eval_grep: "^cifar10_variant_fid\\["

## Baseline (locked)

# Measured on prior runs, CIFAR-10 10K-FID at 150ep, 3 seeds, same backbone
# (FM UNet @ 128ch) and sampler (Euler, 100 steps, c(t)-rescaled velocity):
#   vanilla EqM   (cifar_seed_study):       ~9.5  (5K), ~9-10 (10K mean)
#   DG-ANM v01    (cifar_dganm_ext e150):   21.72 ± 9.9 (10K, n=3)
#
# The pilot proxy used here is the CHEAPER *25-epoch* CIFAR-10 10K-FID. This
# will NOT match the 150-ep numbers in absolute terms — both vanilla and v01
# will be higher. What we need is relative ordering: a variant is promoted
# only if it beats v00_vanilla at 25ep by >=1 FID. Absolute vanilla-comparable
# FID is measured only on promoted variants at the confirmation stage (full
# 150ep, 3 seeds).

baseline_variant: v00_vanilla
reference_variant: v01_current   # the known-broken DG-ANM to beat

## Constraints

max_runtime_seconds: 10800      # 3h walltime per pilot (25ep CIFAR @ ~90s/ep + FID)
max_slurm_minutes: 180

# The dispatcher + shared infra are IMMUTABLE to the autoresearch agent.
# Only variant files and their configs are fair game.
files_allowed:
  - projects/diff-EqM/experiments/dganm_variants/v*.py
  - projects/diff-EqM/configs/variants/v*.json
files_readonly:
  - projects/diff-EqM/program.md
  - projects/diff-EqM/experiments/dganm_variants/_common.py
  - projects/diff-EqM/experiments/dganm_variants/__init__.py
  - projects/diff-EqM/experiments/run_variant.py
  - projects/diff-EqM/slurm/jobs/variant_pilot.sbatch
  - projects/diff-EqM/slurm/jobs/compute_in100_reference_stats.sbatch
  - projects/diff-EqM/eqm-upstream/
  - projects/diff-EqM/fm-upstream/
  - projects/diff-EqM/results/cifar10_inception_stats.npz

partition: gpu
pilot_epochs: 25
confirmation_epochs: 150
num_fid_samples_pilot: 5000
num_fid_samples_confirmation: 10000

## Variant slate (initial)

# Each variant is a concrete reformulation of DG-ANM with a literature citation.
# Filenames: projects/diff-EqM/experiments/dganm_variants/<variant>.py
# The variant is a single self-contained file that exposes train(args).

initial_variants:
  v00_vanilla:
    purpose: "Sanity / baseline. Plain EqM loss, no mining. Must match
              train_cifar_eqm_unet.py within seed noise."
    citation: "—"
  v01_current:
    purpose: "Frozen reference of the broken DG-ANM. Every new variant must beat this."
    citation: "—"
  v02_score_repulsion:
    purpose: "Two-sided cosine contrastive on velocity; drops hinge entirely."
    citation: "Jiang et al. 2025 (VeCoR, arXiv 2511.18942); arXiv 2505.21742"
  v03_noised_negatives:
    purpose: "Mine at t ~ U(0,1) instead of hardcoded t=1. Minimal fix to
              isolate the t-schedule effect while keeping the hinge."
    citation: "Luo et al. 2023 (DCD, arXiv 2307.01668); Du 2021"
  v04_ebm_contrastive:
    purpose: "InfoNCE-with-one-negative on a velocity-norm energy; replaces
              absolute hinge with a relative objective that cannot saturate."
    citation: "Du, Li, Tenenbaum, Mordatch (ICML 2021, CD); DCD 2023"
  v05_drop_geometry:
    purpose: "Pure ablation: v01 minus the feature-normal/tangent projector."
    citation: "Pidstrigach (NeurIPS 2022)"
  v06_diffusion_recovery:
    purpose: "Penalize short reverse-ODE reconstruction error at perturbed
              x_t. Self-calibrating, no margin, no saturation."
    citation: "Kim et al. 2024 (CTM, arXiv 2310.02279); Kong et al. 2024 (ACT-Diffusion)"

## Strategy

strategy: |
  1. Round 1 (baseline pinning): submit v00_vanilla and v01_current at
     pilot scale with 3 seeds each. Both must produce valid FID and their
     relative ordering must match the 150ep paper-scale result (vanilla
     beats v01). This validates the 25-ep proxy. If the ordering flips,
     stop and raise the pilot epoch count.
  2. Rounds 2+: pick next variant from initial_variants (order: v03, v05,
     v02, v06, v04 — cheapest/highest-probability first). Submit 1 seed at
     pilot scale.
  3. Promotion rule: if pilot 1-seed FID is within 2.0 of v00_vanilla, run
     2 additional seeds at pilot scale. If 3-seed mean beats v00_vanilla by
     >=1.0 FID, promote to confirmation (150ep, 3 seeds, 10K FID).
  4. Elimination rule: if pilot 1-seed FID is >5.0 FID worse than
     v00_vanilla, eliminate the variant — do not burn more seeds on it.
  5. Confirmation rule: a variant is "working" only when its 150ep 3-seed
     mean 10K FID beats v01_current by >=1 FID AND is within 3 FID of
     vanilla. Only then is it eligible for Stage B (ImageNet-100).
  6. After initial slate exhausted: propose new variants informed by
     results.tsv + the lit review. Each proposal must cite a paper.
  7. ONE change per variant. If a variant needs to change, make a new
     file (e.g. v02b_score_repulsion_nopgd.py), don't mutate history.

## Ratchet Rules

keep_threshold: 1.0             # pilot FID must beat v00_vanilla by >=1.0
eliminate_threshold: 5.0        # pilot FID >5.0 worse => eliminate variant
revert_on_crash: true
parallel_candidates: 3          # 3 variants * 3 GPUs concurrent

## Termination

max_iterations: 20              # 20 variants max before stopping
max_wall_hours: 60
stop_on_plateau: true
plateau_window: 5               # 5 consecutive variants with no improvement

## Execution Mode

mode: slurm

## Notes

# Proxy = CIFAR-10 25ep 10K-FID. ~90s/epoch on 1 A100 => ~40 min train +
# ~15 min FID => ~1h total per pilot seed. 3 variants x 3 seeds = 9 pilot
# jobs ~= 3h wall-clock on 3 GPUs (or 9h on 1 GPU).
#
# IN-100 is explicitly NOT part of the pilot proxy. The prior 2ep IN-100
# proxy (superseded by this file) gave noisy signals that didn't transfer
# to 80ep IN-100 FID. Until we have a CIFAR variant that beats vanilla,
# IN-100 is off the table.
#
# When adding a new variant: (1) write v{NN}_<name>.py implementing
# step_fn + train(args), (2) add configs/variants/v{NN}_<name>.json,
# (3) add an entry to initial_variants above with citation + purpose,
# (4) submit via variant_pilot.sbatch with CONFIG_PATH set to the new
# config. The shared harness (_common.py, run_variant.py) should need
# zero changes to accept a new variant.
#
# The old autoresearch hyperparameter history is preserved in
# projects/diff-EqM/results_hp_archive.tsv for reference.
