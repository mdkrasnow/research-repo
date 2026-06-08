# v17 CIFAR Morphism-Discovery Gym — Report

**Date:** 2026-06-08
**Code:** `projects/symmetry-discovery/experiments/v17_cifar_gym.py` (reuses morphisms/anchor/AE/policy/
discover from `projects/diff-EqM/experiments/dganm_variants/_multi_morphism.py`)
**Results:** `projects/symmetry-discovery/results/cifar_gym/*.json`
**Compute:** CPU-local, no FID, no pretrained semantic encoders, GT factors unused.

---

## 1. Why this exists

The prior CIFAR bridge was a **weak negative**: visible and anchor were both ordinary CIFAR, so any
on-manifold-moving op (incl. the decoy `big_shear`) looked equally good and discovery degenerated to
`scale`/`rotate`. The v17 synthetic gym worked because visible was a *narrowed* view and anchor was the
*broader valid* distribution, so the gap **pinned** which morphism family was needed. This script
reconstructs that visible→anchor gap on real CIFAR and asks: with a real gap, does discovery recover the
gap-closing family and reject decoys?

## 2. Tasks (visible→anchor gap construction)

| task | visible | expected valid | recoverable? |
|---|---|---|---|
| centerzoom_to_full | center-zoomed (objects enlarged+centered) | scale, translate | **NO** (per-image crop destroys context) |
| lowcolor_to_full | desaturated (chroma×0.25) | saturate, hue, bright | **YES** |
| position_to_full | shifted to corner (+0.25,+0.25) | translate, scale | marginal |
| no_rotation_control | pure translate (dx=0.25) | translate (NOT rotate) | marginal |
| decoy_pressure | = lowcolor visible | saturate, hue, bright | **YES** |

## 3. Calibration (before any payoff)

Per-family probe: apply each family at signed magnitudes `±{0.4,0.7,1.0}`, report best visible→anchor ED
improvement (`delta_ed`, more-negative = better at closing the gap). Pass = visible differs from anchor
AND expected-valid closes the gap more than any decoy.

| task | gap (ED) | best valid Δ | best decoy Δ | separable | PASS |
|---|---|---|---|---|---|
| centerzoom_to_full | 0.434 | scale −0.160 | **big_shear −0.297** | ✗ | **FAIL** |
| lowcolor_to_full | 0.656 | **saturate −0.301** | 0.000 | ✓ | **PASS** |
| position_to_full | 0.087 | translate_y −0.011 | 0.000 | ✓ | PASS (weak) |
| no_rotation_control | 0.087 | translate_x −0.009 | 0.000 | ✓ | PASS (weak) |
| decoy_pressure | 0.656 | **saturate −0.301** | 0.000 | ✓ | **PASS** |

**Anchor-weakness honestly reported:** on `centerzoom`, the conv-ED anchor is **gamed by `big_shear`**
(−0.297, larger than any valid family). Center-zoom is also information-destructive (cropping cannot be
inverted per-image), so it has no true valid recovery. → correctly flagged FAIL, no payoff run.

**Headroom check:** only `lowcolor`/`decoy_pressure` have real oracle headroom (oracle ED 0.584 < base
0.656). `position`/`no_rotation` gaps are tiny (0.087) and reflection-padding artifacts mean even ORACLE
raises ED (0.129 > 0.087) → no payoff headroom; reported, not pursued.

## 4. Discovery — the objective bug and the fix

**First discovery run (original `discover` reward = `−max(conv_ED, 50·AE_recon)`):** collapsed to
**`scale` (0.79)** on *every* task and **raised** ED (lowcolor DISCOVERED 0.847 > base 0.656), despite
`decoy_usage≈0.002`.

Root cause (two compounding):
1. **AE veto mis-fires.** The robust-AE was trained on a nuisance set with hue/bright but **not
   saturation** → it scores `saturate` (the correct family) as *off-manifold* (high recon) and the
   `max()` reward then *penalizes the right family*, while `scale` (in the AE's affine nuisance set)
   scores as perfectly valid.
2. **Validity ≠ gap-direction.** The reward measures *on-manifold-ness*, not *closing the visible→anchor
   gap*. `scale` is on-manifold but irrelevant to a chromatic gap.
   (The `move`-hinge was inert — `move`≈3–4 ≫ margin 0.6 always — so dropping it was a no-op; verified.)

**The 1 retune — gap-aware reward** (`gap_aware=True`): bandit reward = `−conv_ED` only (drop the AE
veto from the reward; keep AE as a mild `×5` regularizer). This is exactly the signal calibration's
`delta_ed` uses.

## 5. Result (gap-aware) — STRONG POSITIVE

`lowcolor_to_full`, seed 0:

| arm | payoff ED (↓) | note |
|---|---|---|
| RANDOM_WITH_DECOYS | 0.750 | worst — decoys hurt |
| RANDOM_VALID | 0.684 | uniform over valid |
| BASE | 0.656 | identity |
| ORACLE_VALID | 0.584 | uniform over {saturate,hue,bright} |
| **DISCOVERED** | **0.398** | **saturate 0.97, decoy_usage 0.008** |

Discovered family weights: **saturate 0.97** (expected), top decoy 0.012. ED trace 0.87→0.40 during
discovery. **DISCOVERED beats BASE, RANDOM_VALID, RANDOM_WITH_DECOYS — and even ORACLE_VALID** (a tuned
single correct family beats a uniform mix that dilutes with hue/bright).

`decoy_pressure` (twin): saturate 0.98, decoy_usage 0.001, DISCOVERED 0.403 vs base 0.656.

### Success conditions (lowcolor)
- ✓ decoy usage < 25% (0.8%)
- ✓ DISCOVERED improves visible→anchor ED over BASE (0.40<0.66) and RANDOM_WITH_DECOYS (0.40<0.75)
- ✓ DISCOVERED beats RANDOM_VALID (0.40<0.68)
- ✓ discovered family matches expected (saturate ∈ expected)

### 3-seed reproducibility (lowcolor, gap-aware) — all strong positive

| seed | saturate wt | decoy_usage | BASE | ORACLE | RND_VALID | RND_DECOY | **DISCOVERED** |
|---|---|---|---|---|---|---|---|
| 0 | 0.97 | 0.008 | 0.656 | 0.584 | 0.684 | 0.750 | **0.398** |
| 1 | 0.965 | 0.018 | 0.682 | 0.602 | 0.701 | 0.768 | **0.424** |
| 2 | 0.987 | 0.003 | 0.719 | 0.644 | 0.736 | 0.812 | **0.437** |

Every seed: discovered = `saturate` (expected), decoy_usage < 2%, DISCOVERED beats BASE, RANDOM_VALID,
RANDOM_WITH_DECOYS **and ORACLE_VALID**. Robust, not a seed artifact.

## 5b. EqM-lite generative payoff (the rung to a real EqM claim)

ED is an anchor metric. The next rung: train a `TinyEqM` (flow velocity-matcher) on the DESATURATED
(visible) split with each augmentation arm, eval EqM loss on FULL CIFAR (`eqm_full`, lower = better
generalization to the missing-color target). `--eqm_lite` flag.

| seed | BASE | ORACLE_VALID | RANDOM_VALID | RANDOM_WITH_DECOYS | **DISCOVERED** |
|---|---|---|---|---|---|
| 0 | 0.3561 | 0.3426 | 0.3431 | 0.3448 | **0.3304** |
| 1 | 0.3589 | 0.3465 | 0.3501 | 0.3498 | **0.3357** |

DISCOVERED is best every arm, both seeds — the ED win **transfers to a generative objective**, not just
the anchor metric. Margins are small (~0.025 below base, ~0.012 below oracle) but consistent. This is the
prediction the real-EqM constructed-gap bridge (`v15_gap_morphism_aug`, desaturated-train CIFAR, FID vs
full) is testing on the cluster.

## 6. Verdict

**CIFAR discovery is REOPENED — on a properly-constructed visible→anchor gap, with a gap-aware reward.**

Two findings the prior weak-negative missed:
1. **Task construction matters.** A real gap (lowcolor) makes the gap-closing family *uniquely*
   rewarded; the prior CIFAR setup (no gap) and destructive gaps (centerzoom) do not, and the conv-ED
   anchor is gameable by `big_shear` there.
2. **Objective matters.** The original reward (on-manifold validity + AE veto) is *not* gap-aware and
   the AE veto can penalize the correct family; switching the bandit reward to ED-to-anchor recovers
   `saturate` cleanly and beats random + oracle.

This is the **first strong positive on natural images** in the v13–v17 arc: discovery recovers the
correct morphism, avoids decoys, and beats both random and oracle on the calibrated payoff proxy.

**Scope/caveats:** validated on a *constructed* chromatic gap (lowcolor) with an EqM-lite-adjacent ED
payoff proxy, single dataset (CIFAR), short discovery. Spatial gaps (position/no_rotation) lack payoff
headroom in this setup; centerzoom is anchor-weak. NOT an FID claim — FID stays human-gated.

## 7. Best next experiment

Port `gap_aware` reward into the EqM bridge variant (`v14_multi_morphism_aug.py`) on a **constructed**
CIFAR gap (train EqM on a desaturated/narrowed visible split, anchor = full CIFAR) rather than full-vs-
full — and test whether discovered `saturate` aug improves EqM-lite / (human-gated) FID *over* random
and known crop **on that gap**. The whole-arc thesis predicts discovery's payoff appears only where the
useful symmetry is *not* already known (crop) — a constructed gap is where it should win.

## 8. Reproduce

```bash
# calibration (all tasks)
python projects/symmetry-discovery/experiments/v17_cifar_gym.py --stage calibrate --n 768 --seed 0
# discovery+payoff, gap-aware (the fix), headroom tasks
python projects/symmetry-discovery/experiments/v17_cifar_gym.py --stage all --n 768 --seed 0 \
    --steps 300 --gap_aware --tasks lowcolor_to_full,decoy_pressure
# original (buggy) reward for contrast: drop --gap_aware
```
