# v17 → real-EqM bridge spec (ready to run on human approval)

**Status (2026-06-07):** bridge designed + cluster session verified LIVE (reached `squeue`; jobs
`exp1-sam`, `eqm-1k-v` already running). NOT submitted. Two hard gates remain, both the human's:
(1) explicit approval to run an FID-scale EqM job (FID is NEVER auto-authorized), and (2) greenlight to
modify `projects/diff-EqM/...` (cross-project; this write-up stays in symmetry-discovery).

## Why the existing bridge code must be upgraded

`projects/diff-EqM/experiments/dganm_variants/v12_stable_generator_aug.py` (and v13) implement the OLD
single-operator recipe: discover ONE affine generator `M = matrix_exp(A)` against a RAW random-conv feature
anchor, then orbit-augment EqM. v17 shows two reasons that under-delivers and what to change:

| dimension | old bridge (v12/v13) | v17-validated recipe |
|---|---|---|
| operator | single affine generator `exp(A)` | MULTI-family policy `q_θ(family,magnitude,composition)` over candidate primitives |
| anchor | raw random-conv energy distance | PCA-WHITENED random-conv ED + appended chroma/edge stats (whitening was the missing piece; raw dims dominate) |
| family selection | n/a (one operator) | EMA-reward bandit over candidates INCLUDING decoys → learns WHICH morphisms are valid, avoids invalid |
| validity signal | stability reg (det≈1, cond→1) | on-manifold ED + move; decoys raise ED → down-weighted (ablation-confirmed load-bearing) |

## The port: `v14_multi_morphism_aug` (new variant, on human go)

Stage 1 (offline, before EqM; reuse v17 `v17_policy.discover` + `AnchorScorer` logic, ported to operate on
CIFAR pixels):
- candidate families = {translate_x, translate_y, rotate, scale, hue, bright} + decoys {crop_erase,
  big_shear, color_collapse} as a negative-control set.
- anchor = PCA-whitened random-conv ED (+ chroma/edge stats) fit on real CIFAR.
- discover a FROZEN multi-family policy; save family weights + magnitudes + `operator_diag.json`.

Stage 2 (EqM training, unchanged loop):
`L = eqm_loss(model, x) + lam_aug * eqm_loss(model, frozen_policy(x).detach())`.

Arms (one table, controlled — reuse the bridge harness + `variant_pilot.sbatch`):
- `v00_vanilla` BASE / floor
- `v10_hard_example` HARDNEG negative
- `vK_known_aug` (flips/crops) KNOWN positive reference
- `v14_multi_morphism_aug` discovered=treatment, and a `random`/`with_decoys` negative control
Metric: FID (5K) + the discovery diagnostics (decoy-usage ≈ 0, on-manifold rate) — trust diagnostics over
FID alone per the coverage/coherence confound.

## Pre-registered gate (before any scale-up)

Pass = discovered FID < random-valid FID AND discovered ≤ known-aug FID AND decoy-usage ≈ 0 at CIFAR.
Only then consider IN-1K. This mirrors the v17 EqM-lite gate that passed.

## Blockers / required human actions (in order)

1. **Approve an FID-scale EqM run** (mandatory; FID never auto-authorized). Until then this stays a spec.
2. **Greenlight editing `projects/diff-EqM/`** to add `v14_multi_morphism_aug` (cross-project; current task
   scope is symmetry-discovery).
3. If a fresh cluster login is needed later: `! scripts/cluster/ssh_bootstrap.sh` (2FA). Session is live now,
   so this may not be needed immediately.

## What is already done toward the bridge

- v17 recipe validated end-to-end (Phase 0–3, 3 datasets) — `documentation/v17_morphismgym_writeup.md`.
- Reusable, ported-ready code: `experiments/v17_policy.py` (`discover`, bandit, grouped apply),
  `v17_morphism_gym.py` (`AnchorScorer` PCA-whitened ED + stats, morphism families, decoys).
- Bridge harness + configs exist in `projects/diff-EqM/configs/variants/bridge/`; the new variant slots into
  the same `variant_pilot.sbatch` flow.

**Next concrete step on approval:** add `v14_multi_morphism_aug.py` to `dganm_variants/`, a
`bridge150_v14_discovered.json` config, smoke it (loss finite + ≥16-sample probe per the diff-EqM
smoke rule), then submit the 4-arm CIFAR bridge — NOT IN-1K, NOT before the gate passes.
