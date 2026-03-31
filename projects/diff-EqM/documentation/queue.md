# Experiment Queue — DG-ANM for EqM

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
