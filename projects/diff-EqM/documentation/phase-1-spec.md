# Phase 1 Spec — v10-only IN-1K seed 0 (RETROSPECTIVE)

Status: **COMPLETE** 2026-05-27. Gate PASS.

This spec is written retrospectively. Phase 1 was executed under live conditions following the CAFM-EqM catastrophe (2026-05-23, FID 341.25) and the resulting Branch B-Both retirement → v10-only pivot. The original phase 1a/1b structure (CAFM-only + CAFM+v10) was retired; what executed was v10-only single-seed validation at paper-comparable scale.

## Objective (as executed)
Confirm that v10 (PGD hard-example mining on EqM regression target) transfers from CIFAR-10 Phase 0.3 PASS (FID 13.40 vs vanilla 14.17) to ImageNet-1K-256 at paper-comparable scale (EqM-B/2, 80 epochs).

## Pre-registered gate
**FID ≤ 30.41** on 50K-sample evaluation (vs trusted vanilla baseline 31.41, threshold = vanilla − 1.0 per CLAUDE.md).

## Result
- **FID = 29.0085** — PASS by 1.40 below threshold, 2.40 below vanilla.
- Training elapsed 1d11h32m (resumed from ckpt_65000 after 15290932 quota-deadlock incident; see debugging.md).
- All mandatory diagnostics green throughout: aux/base ratio 1.006–1.034 (non-saturating), base loss 10.28 matching vanilla 10.32, ‖δ‖=0.300 stable at boundary.
- Smoke-probe FID 78.91 at ckpt_65000 (~13ep) confirmed healthy trajectory mid-train (per CLAUDE.md mandatory sample probe).

## Configuration (frozen)
- Model: EqM-B/2
- Dataset: ImageNet-1K-256 class-conditional
- Epochs: 80 (405K steps at GBS=256)
- Hardware: 4× A100-80GB DDP, seas_gpu
- Train sbatch: `slurm/jobs/imagenet1k_80ep_v10.sbatch`
- Train script: `experiments/train_imagenet.py` (trusted path that produced vanilla 31.41)
- v10 hyperparameters: λ=0.1, K=1, ε_rad=0.3, mining_lr=0.05, mine_every=1
- Sampler: gd (gradient descent), eta=0.003, num_sampling_steps=250, cfg_scale=1.0
- FID: 50000 samples, pytorch-fid against 50K shuf-subsampled IN-1K reference

## Jobs (canonical)
| Job ID | Phase | Outcome |
|---|---|---|
| 15067741 | seed 0 train initial (gpu) | CANCELLED — 12h+ queue pending, no ETA |
| 15290932 | seed 0 train v2 (seas_gpu) | WEDGED at step 74,200 — home quota deadlock |
| 15638767 | seed 0 train resume from ckpt_65000 | COMPLETED clean 1d11h32m, step 405K |
| 15635872 | smoke-probe FID at ckpt_65000 (seas_gpu) | FID 78.91 PASS — healthy trajectory |
| 15933157 | ckpt auto-pruner watchdog (shared) | COMPLETED — kept quota stable 30-50G |
| 16304913 | Phase 1 gate FID (gpu) | CANCELLED — queue depth 647 |
| 16327377 | Phase 1 gate FID v2 (gpu_requeue) | FAILED 2m27s — .JPEG case-sensitivity bug |
| 16328965 | Phase 1 gate FID v3 (gpu_requeue) | COMPLETED FID 29.0085 |

## Downstream
- Triggered Phase 2 launch (seeds 1+2: 16362498, 16362499).
- Triggered Phase 3 baseline launch (vanilla S/2, L/2: 16369650, 16369651).
- Triggered Phase 3 v10 scaling smokes (S/2, L/2, XL/2).
- PI update drafted (`pi-updates.md` 2026-05-27 entry) — user-send only.

## Postmortem-style lessons
1. **Pre-staged helpers must reference trusted-baseline sbatch paths.** The pre-staged `submit_v10_phase1_fid.sh` initially pointed to the generic `imagenet_fid.sbatch` (IN-100 lineage, case-sensitive `.jpeg`), not the IN-1K-specific `imagenet1k_fid_eval.sbatch` that produced the trusted vanilla 31.41. Burned 2m27s + one retry cycle. Fix landed `fef80d7`.
2. **Quota-deadlock from rsync-of-ckpts is a recurring failure mode.** Manual pruning kept this one alive but two near-misses (15290932 wedge, 15638767 quota creep) confirm the underlying bug. Auto-pruner sbatch (`prune_v10_ckpts.sbatch`) was deployed mid-run and worked. Underlying fix (thin /tmp side before rsync) deferred to next sbatch revision.
3. **Train completed despite mid-run quota crisis.** Resume-from-ckpt + auto-pruner combination is robust.
