# Full-CIFAR v10-ANM Comparison Ladder (Track B) — pre-registration

**Date:** 2026-06-08
**Question:** Can discovered morphisms become a stronger augmentation policy than the hand-designed v10
ANM lever on the SAME base task — or at least ADD lift v10 lacks (complementarity)?

**Framing correction (load-bearing):** crop/flip are a **standard strong CIFAR default, not an oracle.**
"Discovery has no headroom because crop is near-optimal" is RETRACTED as too broad. Discovery wins if it
beats the hand-designed lever (v10) OR adds lift on top of it — not only if it dethrones crop.

## Protocol (identical across arms)
- Base task: full CIFAR-10, EqM UNet (`build_unet`), 150 epochs, lr 2e-4, bs 128.
- Eval: FID(ema) every 50 ep (2000 samples), final 5000 samples, Euler 100 steps.
- Same harness (`run_variant.py` / dganm_variants) for ALL arms — NO cross-harness comparison
  (documented 4.7-FID variant↔legacy gap; only same-harness numbers are valid).
- Seeds: start seed 0 (single-seed screen); promote winner to 3 seeds for a Welch test.

## Pre-registered arms

| Arm | variant | config | purpose |
|---|---|---|---|
| v00 base | v00_vanilla | ladder150_v00.json | no-aug control |
| v10 ANM | v10_hard_example | ladder150_v10.json | strong hand-designed lever |
| v17 random | v14_multi_morphism_aug (random) | ladder150_v17rand.json | controls for "more aug" |
| v17 discovered | v14_multi_morphism_aug (discovered) | ladder150_v17disc.json | discovery alone |
| **v10+v17 hybrid** | v16_hybrid_mine_morph (discovered) | ladder150_hybrid.json | **complementarity (likely real win)** |

## Staged success (pre-registered)
- **Minimum:** v17 discovered < v17 random (discovery does something real beyond sampling augs).
- **Strong:** v17 discovered < v00 base AND approaches v10.
- **Major:** v17 discovered < v10.
- **Best publishable:** **hybrid < v10 alone** (discovery adds lift the lever lacks).

## Existing same-harness hints (need clean same-seed rerun to trust)
crop 12.59 < v10 13.40 < v17-disc 14.21 < v00 14.31 < v17-rand 14.59 (5K).
→ discovery currently LOSES to v10. So the load-bearing runs are the **hybrid** and, if discovery loses,
a **diagnostic**: is v17 picking weak-but-valid morphisms, wrong magnitude, under-weighting spatial
transforms, over-favoring hue/bright, or raising EqM loss? (Full-CIFAR v14 discovery favors
translate/scale/hue — valid but maybe not highest-value for the generator.)

## RESULTS — seed 0 (2026-06-09, 5K FID, full CIFAR, same protocol)

| arm | FID (5K) | vs v10 |
|---|---|---|
| **hybrid (v10+disc)** | **13.15** | −0.89 |
| v17rand | 13.39 | −0.65 |
| v17disc | 13.47 | −0.57 |
| v10 ANM | 14.04 | — |
| v00 base | 14.31 | +0.27 |

Staged gates: MIN (disc<random) **FAIL** (13.47>13.39); STRONG (disc<base) PASS; MAJOR (disc<v10) PASS;
**BEST (hybrid<v10) PASS** by 0.89.

**Interpretation (seed 0, single-seed — margins need 3-seed confirmation):**
- **Discovery alone ties random** (13.47 vs 13.39). On full CIFAR (no gap) the validity reward picks
  whatever is on-manifold; v17disc landed SPATIAL (scale 0.35 + rotate/translate, spatial_mass 0.77) =
  crop-like, so it ≈ random-valid. Expected: no gap → nothing unique to discover.
- **Discovery is unstable run-to-run on full CIFAR**: hybrid's policy went PHOTOMETRIC (effective saturate
  0.35, hue 0.18) while v17disc went spatial — same config, different families. No gap to pin the policy.
- **All morphism arms beat v10 (14.04); hybrid lowest (13.15).** BUT v10 came in 0.64 ABOVE its prior-run
  hint (13.40) — single-seed v10 is noisy/unlucky, so "beats v10" is partly v10's seed. The 0.89
  hybrid<v10 margin is the most robust but still single-seed.
- Cosmetic flag: hybrid put 0.29 WEIGHT on color_collapse at ~0 magnitude (effective decoy 0.022 —
  functionally a no-op); the "keep-lowest-effective-decoy" restart selection allows high-weight/zero-mag
  decoys. Harmless here, worth tightening.

**Verdict:** Track B is INCONCLUSIVE at single-seed (all top arms within v10's own ±0.6 seed swing). The
hybrid-lowest + all-morphism-beats-v10 direction is promising but NOT a result without seeds. Do NOT tune
HP on this noise. NEXT = 3-seed Welch on {v10, hybrid} (and gap15 {discovered,known} for the flagship).

## Decision rule
- hybrid < v10 (seed 0) → promote hybrid to 3 seeds, Welch t. If holds → publishable complementarity.
- v17 disc < v10 (seed 0) → promote v17 disc to 3 seeds.
- All discovery arms ≥ v10 and hybrid ≥ v10 → discovery does not beat/augment the lever on full CIFAR;
  fall back to the Track-A constructed-gap claim (where discovery's value is real and demonstrated).
