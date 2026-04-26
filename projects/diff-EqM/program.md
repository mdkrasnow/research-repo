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

max_runtime_seconds: 21600      # 6h walltime per pilot (100ep CIFAR @ ~110s/ep + FID; mining variants ~2x)
max_slurm_minutes: 360

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
pilot_epochs: 100
confirmation_epochs: 150
num_fid_samples_pilot: 5000
num_fid_samples_confirmation: 10000

# History: pilot_epochs was 25 in the first attempt at Round 1 (jobs 8370711,
# 8370723). At e25 both vanilla and v01 are still in the steep descent phase
# (FID ~340), and v01's hinge is provably inert at random init for the first
# few epochs (neg_loss=0 on every logged epoch). The 25ep proxy could not
# resolve the known 150ep vanilla<<v01 gap (vanilla 12.54 vs v01 21.66 at
# 150ep, 10K FID). Bumped to 100ep based on the prior 3v3 seed study, which
# showed clean separation at e100 (vanilla 30.18 +- 0.28 vs DG-ANM
# 43.83 +- 7.62, Welch t~3.1σ).

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
  1. Round 1 (vanilla noise-floor pinning): submit v00_vanilla x 3 seeds
     at 100ep pilot. This produces (mu_vanilla, sigma_vanilla) — the
     reference distribution every new variant is compared against. v01 is
     NOT piloted: its 150ep deficit is already established at 3 seeds, and
     re-running it at the proxy length adds zero information.
  2. Rounds 2+: pick next variant from initial_variants (order: v03, v02,
     v06, v04, v05 — most-likely-to-have-signal first; v05 is an ablation
     and likely a null result, scheduled last). Submit 1 seed at pilot.
  3. Promotion rule (z-test against vanilla noise floor): if pilot 1-seed
     FID < mu_vanilla - 1.96 * sigma_vanilla - 1.0, run 2 more seeds at
     pilot. If 3-seed mean beats mu_vanilla by >= 1.0 FID with
     (mean + sigma) below mu_vanilla, promote to confirmation
     (150ep, 3 seeds, 10K FID).
  4. Elimination rule: if pilot 1-seed FID is >5.0 FID worse than
     mu_vanilla, eliminate the variant — do not burn more seeds on it.
  5. Confirmation rule: a variant is "working" only when its 150ep 3-seed
     mean 10K FID beats vanilla's 150ep 3-seed mean (12.54 +- 1.15) by
     >= 1 FID. Only then is it eligible for Stage B (ImageNet-100).
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
