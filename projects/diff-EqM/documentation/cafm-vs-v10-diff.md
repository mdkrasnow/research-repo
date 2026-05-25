# CAFM Phase 1b → v10 Phase 1 — What Changed

Quick-reference diff for what we changed after the CAFM-on-EqM failure
(postmortem `postmortem-cafm-eqm-2026-05-23.md`) when pivoting to v10-only.

## Method-level diff

| Aspect | CAFM (FAILED, FID 341.25) | v10 (current run, 15290932) |
|---|---|---|
| Method | Adversarial post-training of vanilla ckpt + discriminator + JVP | PGD hard-example mining on EqM regression target |
| Loss form | LSGAN on dis logits + centering-penalty regularizer | `MSE(f(x_t+δ*), target)` where `δ* = argmax MSE` over `‖δ‖₂ ≤ ε` |
| Two-player? | Yes (gen vs dis adversarial) | No (single model, single objective) |
| Can collapse to trivial? | Yes (gen → 0 magnitude to fool dis) | **No** (bounded by EqM target; aux loss can't be negative) |
| Discriminator | Fresh DiT-B/2, randomly initialized | None |
| Start point | Vanilla ckpt 380K (post-training) | Scratch (80ep from random init) |
| Comparable to vanilla baseline? | No (different recipe, post-trained) | **Yes** (same recipe + epochs as vanilla 31.41) |
| Mining ratio behavior | N/A | Matches CIFAR Phase 0.3 PASS (1.02 vs 1.05) |
| Observed loss curve | Dis crushed gen 2.0 → 0.03 one-sided | Base loss stable 10.5 (matches vanilla 10.3) |
| sbatch | `cafm_eqm_b2_in256.sbatch` (4×A100, 48h) | `imagenet1k_80ep_v10.sbatch` (4×A100, 48h) |
| Training script | New `train_cafm_eqm.py` (fresh code, manual DDP all_reduce, multiple infra bugs) | Existing `train_imagenet.py` (proven path that produced vanilla FID 31.41), v10 branch added |
| Pre-validation | Smoke validated loss-finiteness only | CIFAR Phase 0.3 PASS validated FID improvement + loss-curve shape |

## Structural lessons from postmortem now in place

1. **Mandatory smoke-time sample probe** (`CLAUDE.md` "Smoke-time sample probe"
   section). `ckpt-every=5000` lets us FID a 5K-step ckpt before letting full
   80ep complete.
2. **Adversarial-loss oscillation check** (`CLAUDE.md` "Discriminator-based
   loss check"). One-sided monotonic dis-loss decrease = STOP, not retune.
3. **Trusted-path training script.** v10 wired into `train_imagenet.py`
   (which produced vanilla FID 31.41) instead of a fresh trainer with
   parallel infra bugs.

## Critical mechanism distinction

CAFM is **adversarial two-player** with two collapse modes:
- Dis wins → gen forced to fool by suppressing field → "field-collapse" failure mode
- Gen wins → mode collapse

v10 is **single-player regression** with the same loss family the network
was trained on:
- Aux loss = same MSE form as base loss, evaluated on a mined input
- No degenerate minimum (target is fixed, model can only get closer)
- Per Briglia 2025 §3, K=1 FGSM-style adversarial regression is the safest
  variant of input-space adversarial training for regression-target models

Mathematically much safer surface than CAFM. CAFM's collapse to FID 341
in 250 gen updates would not have a v10 analog.

## What CAFM cost taught the project

- ~33 GPU-h spent on CAFM iterations + Phase 1b + diagnostics
- 4 commits adding infra (DDP all_reduce, NCCL/MIG diagnosis, cluster
  sbatch staleness fix, ckpt symlinking) that benefit all future runs
- One mechanism-level fail caught + documented; same class of failure
  caught at smoke time going forward

Postmortem: `postmortem-cafm-eqm-2026-05-23.md`.
Trigger: Phase 1b FID 341.25 vs vanilla 31.41 (10× worse).
Decision: Branch B-Both retired; v10-only pivot.
