# Off-trajectory field robustness diagnostic (Experiment 2)

Compares **vanilla EqM** vs **ANM (v10) EqM** as *field predictors* (not samplers) on
held-out ImageNet validation latents. Tests the mechanistic claim that ANM keeps the
EqM field correct in a local L2 tube around the vanilla noise→data trajectory.

`offtraj_field_robustness.py` — the diagnostic. `../../slurm/jobs/offtraj_field_diag.sbatch` — cluster wrapper.

## What it measures
For data latent `x1`, Gaussian noise `x0`, interpolation `t`, builds the repo's EqM
point and target, then evaluates each model at perturbed points `xt + δ`:

| metric | meaning |
|---|---|
| `mse` | field error vs target `(x1−x0)·c(t)` |
| `cosine` | directional alignment with target |
| `rel_norm_err` / `abs_norm_err` | magnitude calibration |
| `field_norm_rms`, `norm_ratio` | field-norm stability (rules out win-by-rescaling) |

Aggregated by `checkpoint × perturbation_type × radius × t_bin`, plus paired
ANM−vanilla differences with bootstrap CI over `sample_id`.

## Target convention (load-bearing)
Reuses `eqm-upstream/transport` directly — **no hand-coded formula**:
`xt = t·x1 + (1−t)·x0`, `target = (x1−x0)·get_ct(t)`, with `x1`=data, `x0`=noise.
This matches `experiments/train_imagenet.py:_v10_pgd_hard_example_step` (the loss both
checkpoints were trained with). The retired CAFM module `cafm_eqm/eqm_target.py` uses
the OPPOSITE label convention — do not mix.

## Perturbation families
- `random_l2` / `random_l2_orthogonal` — Gaussian δ scaled to a relative radius
  `‖δ‖/‖x1−x0‖`; orthogonal variant projects off the path direction first.
- `sampler_drift` (`--perturbation-type all` or `sampler_drift`) emits two:
  - `sampler_endpoint_mined` — the **real v10 mining** δ (PGA, L2-ball ε=0.3) from a
    frozen shared probe. ε=0.3 absolute ⇒ relative radius ≈ 0.005 in (4,32,32) latents.
  - `sampler_local_drift` — short frozen GD/NAG rollout from `xt`, orthogonalized.

## Gotchas baked in
- `EqM.forward` calls `x0.requires_grad_(True)` internally → use `torch.no_grad()`,
  **never** `inference_mode()`.
- EqM is time-invariant (`uncond` zeros `t` inside forward); `t` still drives the
  target `c(t)` and binning.
- Both checkpoints are plain EqM-B/2 (`ebm=none`, `{model,ema,opt,args}` format);
  EMA weights used by default (matches FID protocol). Primary comparison uses
  matched training step (380000).

## Smoke test
IMPORTANT: comma-separated values (`T_VALUES`, `RADII`) MUST be set as shell env
vars BEFORE `sbatch` with `--export=ALL` — passing them inside the `--export=` list
silently truncates at the first comma (SLURM parses `--export` as comma-separated
KEY=VAL, so `RADII=0,0.1` becomes `RADII=0`).
```bash
GIT_SHA=$(git rev-parse HEAD) \
VANILLA_CKPT=projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt \
ANM_CKPT=projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt \
OUTPUT_DIR=projects/diff-EqM/results/diagnostics/offtraj_smoke \
NUM_BATCHES=1 BATCH_SIZE=8 T_VALUES=0.25,0.75 RADII=0,0.1 \
PERTURB_TYPE=random_l2 PRECISION=fp32 \
  sbatch --export=ALL projects/diff-EqM/slurm/jobs/offtraj_field_diag.sbatch
```
Smoke HARD-FAILS (exit nonzero) if radius-0 cosine ≤ 0 (sign mismatch), radius-0
`xtd ≠ xt`, or any non-finite field. Check `sanity/first_batch_checks.json`.

## Full run
Defaults (no `T_VALUES`/`RADII` override) use the built-in 10-t grid and 7-radius
sweep — safest, since they avoid the comma issue entirely. Override only via shell
env + `--export=ALL` (never inside the `--export=` list).
```bash
V=projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt
A=projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt

# Run 1: random orthogonal sweep (≥5k latents) — uses default t-grid + radii
GIT_SHA=$(git rev-parse HEAD) VANILLA_CKPT=$V ANM_CKPT=$A \
OUTPUT_DIR=projects/diff-EqM/results/diagnostics/offtraj_random \
NUM_BATCHES=80 BATCH_SIZE=64 PERTURB_TYPE=random_l2_orthogonal \
  sbatch --export=ALL projects/diff-EqM/slurm/jobs/offtraj_field_diag.sbatch

# Run 2: ANM-mined + local drift (shared frozen vanilla probe)
GIT_SHA=$(git rev-parse HEAD) VANILLA_CKPT=$V ANM_CKPT=$A \
OUTPUT_DIR=projects/diff-EqM/results/diagnostics/offtraj_sampler \
NUM_BATCHES=80 BATCH_SIZE=64 PERTURB_TYPE=all \
  sbatch --export=ALL projects/diff-EqM/slurm/jobs/offtraj_field_diag.sbatch
```
Add `RESUME=1` to continue an interrupted run (skips completed batches in the jsonl).
Secondary: rerun Run 1 with `ANM_CKPT=<λ=0.3 best-FID ckpt>` — report separately (best-FID, not matched-step).

## Outputs (`OUTPUT_DIR/`)
```
config.json  per_sample_metrics.jsonl  aggregate_metrics.csv  paired_differences.csv
sanity/first_batch_checks.json
plots/{mse,cosine,rel_norm_err}_vs_radius.png
plots/<ptype>__tbin_heatmap_{mse,cosine,relnorm}_diff.png
plots/field_norm_hist_{radius0,offtraj}.png
```

## Interpretation
- **ANM success**: radius-0 parity (cosine/MSE within tolerance) **and** slower
  off-trajectory degradation — lower MSE, higher cosine at nonzero radii, especially
  at the mined relative radius ≈ 0.005; field-norm histograms stable (no collapse/explosion).
- **ANM failure / ambiguous**: better FID but no off-traj field gain; wins only via
  field-norm reshaping; improves MSE but not cosine; only near t≈1 where target norm → 0.
- This is a **local field-robustness proxy** — it does **not** prove global energy
  correctness or sampling optimality.
