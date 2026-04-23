# Experiment Queue — DG-ANM for EqM

## PUBLICATION GOAL
Target: **NeurIPS / ICML / ICLR**. See `documentation/publishability-plan.md`.

Current stage: **A — variant-search autoresearch** on CIFAR-10 25ep pilot. Hyperparameter sweeps abandoned 2026-04-23 after 150ep 3-seed confirmation showed v01 DG-ANM (10K FID 21.66 ± 9.95) loses to vanilla (12.54 ± 1.15) and `diag_dganm_signal.py` proved the margin-hinge is structurally inert.

See `program.md` and `documentation/research-plan.md` for the new plan. Variant slate: v00_vanilla, v01_current, v02_score_repulsion (VeCoR 2025), v03_noised_negatives (Luo 2023 DCD), v04_ebm_contrastive (Du 2021 CD), v05_drop_geometry (Pidstrigach 2022), v06_diffusion_recovery (Kim 2024 CTM).

## Top-of-queue (variant-search autoresearch)
1. **Round 1 baseline pinning**: submit v00_vanilla and v01_current at 25ep pilot, 1 seed each. Must reproduce the known 150ep ordering (vanilla << v01) or raise pilot_epochs. Config: `configs/variants/v{00,01}*.json`. Sbatch: `slurm/jobs/variant_pilot.sbatch` with `CONFIG_PATH` env var.
2. **Round 2 cheap variants**: v03_noised_negatives + v05_drop_geometry (1 seed each). v03 is the highest-probability single-change win per lit (fix t-schedule mining bug).
3. **Round 3 reformulations**: v02_score_repulsion + v06_diffusion_recovery (1 seed each). Both replace the saturating hinge with non-saturating objectives.
4. **Round 4 exotic**: v04_ebm_contrastive (1 seed). Budget for instability per Du 2021.
5. **Promotion gate per variant**: 1-seed pilot beats v00 by >=1 FID → run 2 more seeds at pilot scale. 3-seed pilot mean beats v00 by >=1 FID → promote to 150ep 3-seed confirmation.
6. **Confirmation gate (Stage A.5)**: 150ep 3-seed 10K FID mean beats v01 by >=1 FID AND within 3 FID of vanilla.
7. **Stage B (gated on A.5 passing)**: winning variant vs vanilla on IN-100 EqM-B/2 80ep × 3 seeds, 50K FID. Blocked on IN-100 ref-stats `KeyError: mu` offline fix (3 completed vanilla seed runs preserved, FID recomputable without retrain).

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
