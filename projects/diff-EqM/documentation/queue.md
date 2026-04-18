# Experiment Queue — DG-ANM for EqM

## PUBLICATION GOAL
Target: **NeurIPS / ICML / ICLR**. See `documentation/publishability-plan.md`.

Current stage: **A (proxy sweep + 3-seed repeatability gate)**. Proxy gains (FID ~250+) are NOT publishable on their own — they are a filter for Stage B, where we confirm at EqM-B/2 + CIFAR-10 + 80ep against vanilla EqM. Do not scale up until the Stage A exit gate passes.

## Top-of-queue (derived from publishability plan)
1. **Stage A.5 Step B — DDPM UNet smoke test (chosen path: B->A)**: build a diffusers `UNet2DModel` training loop on CIFAR-10 with plain FM/DDPM loss, reusing our existing data + FID eval pipelines. Exit: FID<15 in a few hours → proceed to Step A. See `documentation/stage-a5-plan.md`. Root cause audit in `documentation/stage-a5-audit.md` (confirmed: wrong architecture, our upstream has no UNet).
1b. **Stage B vanilla IN-256 baseline (start in parallel)**: submit vanilla EqM-B/2 80ep on ImageNet-256 on the known-working transformer stack. Multi-day run; starts the critical-path baseline for Stage B's DG-ANM vs vanilla comparison.
1c. **Stage A.5 Step A — port FM UNet + graft EqM loss (~1 week)**: blocked by Step B passing. Vendor `facebookresearch/flow_matching`'s CIFAR UNet, swap FM loss for EqM `c(γ)`-weighted target, train vanilla to FID ≤3.66. Then DG-ANM variant for Stage C secondary result.
2. Complete round 4 tournament (9 candidates running).
3. Round 5: combination of top dimension winners (tests additivity).
4. **Stage A exit gate**: best config × 3 seeds on the proxy. Gain must exceed seed-std by ≥1 FID.
5. **Stage B (REVISED target)**: DG-ANM vs vanilla EqM on **ImageNet-256 class-conditional, EqM-B/2, 80 epochs, 3 seeds each, 50K-sample FID**. NOT CIFAR — CIFAR is an appendix ablation in the EqM paper and EqM is actually worse than Flow Matching there (3.36 vs 2.09). ImageNet-256 is where EqM is strong and where a reviewer expects the headline result.
6. **Stage C**: scaling curves on ImageNet-256 (S/2, B/2, ±L/2) + CIFAR-10 3-seed secondary (appendix-style, mirroring the EqM paper's own structure).

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
