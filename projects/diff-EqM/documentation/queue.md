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
