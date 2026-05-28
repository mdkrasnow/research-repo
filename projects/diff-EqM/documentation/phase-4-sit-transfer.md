# Phase 4 Spec — v10 → SiT Transfer

Status: PLANNED. Triggers on Phase 3 scaling-curve PASS.

This spec supersedes `phase-5-sit-headtohead.md` (which was written under Branch B-Both framing; the v10+CAFM composition has been retired and Phase 4 is now v10-only transfer).

## Objective
Confirm that **v10 (PGD hard-example mining on regression target)** transfers from EqM to SiT (Ma et al. 2024). This broadens the methodological claim from "EqM-specific" to "regression-target generative models generally."

## Pre-registered gate
- **>= 0.5 FID gain** for SiT-B/2 with v10 vs SiT-B/2 vanilla, at IN-1K-256 80ep, 50K-sample FID.
- If gain >= 0.5: claim broadens to "regression-target gen models." Workshop §4 includes SiT result.
- If gain < 0.5: claim narrows to EqM only. Workshop still publishable on EqM scaling + CAFM postmortem.

## Setup decision

### SiT-B/2 vs SiT-XL/2 with pretrained ckpt
Original `phase-5-sit-headtohead.md` recommended SiT-XL/2 with Lin's pretrained ckpt (`SiT-XL-2-256.pt`) to skip retraining cost.

**Post-pivot decision**: use **SiT-B/2 from scratch**, not SiT-XL/2 pretrained, for these reasons:
1. v10 is a **training-time** modification. Post-training pretrained ckpts is the CAFM playbook (now retired). v10 modifies the train loop — apples-to-apples with vanilla SiT-B/2 from scratch is cleanest.
2. SiT-B/2 80ep IN-1K-256 ~ B/2 EqM 80ep (~150 GPU-h). Affordable.
3. Avoids inheriting whatever Lin's ckpt characteristics are (different recipe, different seed).
4. Direct comparability with our EqM-B/2 80ep results.

### Compute estimate
- vanilla SiT-B/2 80ep seed 0: ~36h × 4 GPU = 144 GPU-h
- v10 SiT-B/2 80ep seed 0: ~50h × 4 GPU = 200 GPU-h (v10 ~50% mining overhead)
- Multi-seed if Phase 4 PASS: +2 seeds × 2 conditions = 4 more runs = ~700 GPU-h total
- Phase 4 full: ~350 GPU-h single seed; ~1000 GPU-h with multi-seed

## Implementation plan

### Code reuse
- Lin's `Adversarial-Flow-Models/models/cafm/sit/sit.py` provides SiT model class.
- Lin's `train_fm_sit.yaml` provides flow-matching training config.
- Our `experiments/train_imagenet.py` v10 hook already implements PGD mining on regression target. Port to SiT with these mods:
  - Replace EqM target `(ε-x)·c(γ)` with SiT/FM target `ε-x` (vanilla FM velocity).
  - Replace EqM model class with SiT model class.
  - Keep all v10 hyperparameters (λ=0.1, K=1, ε_rad=0.3, mining_lr=0.05).
- Existing `experiments/cafm_eqm/v10_sit_trainer.py` is OOP-heavy (subclasses Lin's `ContinuousAdversarialFlowTrainer`). For v10-only (no CAFM dis), simpler is fork our trusted `train_imagenet.py` path.

### Files to write at Phase 4 launch
1. `experiments/train_imagenet_sit.py` — fork of `train_imagenet.py` with SiT model + vanilla FM target. v10 hook unchanged.
2. `slurm/jobs/imagenet1k_80ep_sit_scaling.sbatch` — parameterized like `imagenet1k_80ep_vanilla_scaling.sbatch` but uses SiT model.
3. `slurm/jobs/imagenet1k_v10_sit_scaling.sbatch` — same with `--mining-flavor v10`.

### Smoke gate (per CLAUDE.md)
- SiT-B/2 vanilla 1ep smoke: validate code path + memory + DDP.
- SiT-B/2 v10 1ep smoke: validate mining on FM target + diagnostics.

## Expected timeline
- Trigger: Phase 3 scaling-curve PASS (~2026-06-03 if smokes + S/2/L/2 all land).
- Smokes: ~2h compute, 1 day wall (queue + iteration).
- vanilla SiT-B/2 baseline: ~36h.
- v10 SiT-B/2: ~50h.
- Phase 4 verdict: ~5-7 days post-trigger → roughly 2026-06-10.

## What's already staged
- Lin's SiT code cloned at `external/Adversarial-Flow-Models/models/cafm/sit/`.
- v10 mining loss code in `experiments/cafm_eqm/v10_mining.py` (framework-agnostic; reusable).
- This spec.

## What's NOT yet staged (defer until Phase 3 PASS to avoid premature work)
- `train_imagenet_sit.py` (fork)
- Smoke sbatch
- Full-train sbatch

## Open questions
- Use Lin's SiT-XL/2 ckpt anyway (Phase 4 stretch)? Would require porting v10 as training-time modification on a pretrained backbone — same model-architecture compatibility issues that broke CAFM-EqM. Defer.
