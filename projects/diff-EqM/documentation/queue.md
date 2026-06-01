# Experiment Queue — DG-ANM for EqM

## PUBLICATION GOAL
Targets: **NeurIPS 2026 workshop (Aug 29)** + **ICLR 2027 (~Oct 1)**. See `documentation/summer-2026-plan.md`.

Current phase: **1 — v10-only IN-1K seed 0 train RUNNING** (job 15638767 on seas_gpu, ~10h to completion).
Branch B-Both retired 2026-05-23 after CAFM-EqM Phase 1b FID 341.25 catastrophe (postmortem `postmortem-cafm-eqm-2026-05-23.md`).
v10 = PGD hard-example mining on EqM regression target. Single-objective, no discriminator, mining-based.

## Top-of-queue (Phase 1 → Phase 2)

1. **WAIT** for v10 train 15638767 to complete (step ~285K of 380K at last check; ETA ~10h on seas_gpu, 48h cap). Auto-pruner 15933157 keeps quota in check.
2. **ON COMPLETION** → `bash projects/diff-EqM/experiments/cafm_eqm/submit_v10_phase1_fid.sh` (50K-sample FID on latest ckpt). Gate: **FID ≤ 30.41** (vs vanilla 31.41).
3. **IF GATE PASS** → `bash projects/diff-EqM/experiments/cafm_eqm/submit_v10_phase2_seeds.sh` (seeds 1+2 80ep each). Phase 2 gate: 3-seed Welch t p<0.05, mean ≥ 1 FID gain.
4. **IF GATE FAIL** → 1 retune of λ ∈ {0.03, 0.3, 1.0} per CLAUDE.md, then kill direction → propose v11 (Briglia equivariant fallback, sketch in `documentation/v11_fallback_sketch.md`).
5. **PI update trigger** on Phase 1 gate result (drafted in `pi-updates.md`, user-send only).
6. **Phase 3** (gated on Phase 2 PASS): scaling curves S/2, B/2, L/2 on IN-1K.
7. **Phase 4** (gated on Phase 3): SiT transfer ≥ 0.5 FID.
8. **Phase 5** (gated on Phase 4): workshop draft ready by 2026-08-22 (7-day buffer to deadline).

## Analysis experiments (mechanism / robustness — built, NOT yet submitted)
- **Exp 1 — NFE/sampler robustness** (`experiments/exp1_sampler_robustness/`, README there): vanilla vs v10 B/2 frozen-ckpt sampler sweep {gd,ngd}×nfe{10,25,50,100,250}×step_mult{0.5,1,1.5,2}, FID+KID, AUC + nfe-to-match. Eval-only (no training touched). Local dry-run + synthetic analysis/plots verified. NEXT: Smoke A→B on cluster once ckpt paths confirmed, then 5k pilot, then full 50k via `slurm/jobs/exp1_sweep.sbatch`. Strengthens the workshop sampling-robustness story alongside the FID result.
- **Exp 2 — off-trajectory field robustness** (`experiments/diagnostics/offtraj_field_robustness.py` + `slurm/jobs/offtraj_field_diag.sbatch`, README in `experiments/diagnostics/`): vanilla vs v10 as FIELD predictors. Field MSE/cosine/norm-calibration vs perturbation radius around the EqM interpolation path; random-orthogonal + real-v10-mined + GD-drift perturbations; paired bootstrap CI over sample_id. Reuses repo transport target `(x1−x0)·c(t)` (matches train_imagenet `_v10_pgd_hard_example_step` — authoritative, NOT the retired CAFM eqm_target.py convention). Tests the mechanistic claim behind the FID gain: does ANM make the field more accurate off-trajectory? Pure-logic CPU self-test PASS (target convention, mining L2-ball, orthogonality, metrics, aggregate/paired/CSV). Eval-only, single GPU. NEXT: smoke (1 batch, radii {0,0.1}) — must clear radius-0 sign gate (cosine>0) — then full sweep (80×64 latents) once vanilla+v10 380000.pt ckpt paths confirmed on cluster.

## In-flight
- 15638767 v10 IN-1K seed-0 train (seas_gpu, RUNNING)
- 15933157 ckpt auto-pruner (shared, RUNNING)

Historical top-of-queue items preserved below for reference but no longer active.

## Research Question
Does differential-geometry-guided adversarial negative mining improve EqM's equilibrium landscape, reduce spurious equilibria, and improve optimization-based sampling?

## Hypotheses to Test
1. Baseline EqM-S/2 has measurable short_horizon_recovery_distance on CIFAR-10
2. Normal-space perturbations produce harder negatives than random perturbations
3. Adversarial search (PGA on mining objective) finds harder negatives than random normal-space
4. Trajectory failure term (L_traj) adds signal beyond field norm (L_weak)
5. Combined DG-ANM improves recovery distance vs baseline

---

## READY

### Q-001: Baseline EqM-S/2
- Hypothesis: Establish baseline performance — no mining
- Config: `configs/baseline.json`
- Resources: 1x A100, ~5 min (1 epoch CIFAR-10)
- Priority: HIGH (must run first, establishes baseline metric)
- Notes: This is the autoresearch baseline iteration

### Q-002: DG-ANM Basic (normal + weak)
- Hypothesis: Normal-space perturbations with L_weak improve recovery
- Config: `configs/dganm_basic.json`
- Resources: 1x A100, ~8 min (mining overhead)
- Priority: HIGH (first DG-ANM test)
- Notes: Simplest mining — validates geometry matters before adding complexity

## IN_PROGRESS
(none)

## DONE
(none)

## FAILED
(none)
