---
name: diff-EqM variant search load-bearing findings
description: Critical results and gotchas from DG-ANM variant search on CIFAR-10 (Apr–May 2026). Read before designing more variants or comparing FIDs.
type: project
originSessionId: 9bce0bb7-107d-4741-8833-aa999578b23b
---
## Headline (R5 confirmed, 2026-05-04)

On the **variant harness** (run_variant.py + dganm_variants/_common.py), CIFAR-10 150ep / 5K-FID, 3 seeds each:

| variant | mean ± std | n |
|---|---|---|
| v00_vanilla | **15.06 ± 0.90** | 3 |
| **v02_score_repulsion** | **12.96 ± 0.70** | 3 |
| v07_anneal_v02 | 13.00 ± 0.63 | 3 |
| v08_v02_plus_v06 | 12.90 | 1 |
| v02 λ-sweep (0.1–2.0) | 12.50–12.96 | flat |

**v02 beats vanilla by 2.10 FID (Welch t≈3.2, p≈0.04).** Real, repeatable, statistically significant. Clears program.md keep_threshold (≥1.0) by 2×.

## Load-bearing gotchas

**1. NEVER compare FIDs across the variant harness and the legacy cifar_seed_study runner.**
- Legacy runner's vanilla 150ep ≈ 9.5 (5K FID).
- Variant harness's vanilla 150ep = **14.17** (5K FID), seed 0.
- The 4.7-FID gap is harness-side (dataloader/EMA/sampler diffs), not method-side.
- Why: I spent two cycles concluding "aux losses are net harmful" based on the 9.5 number. Wrong baseline → wrong conclusion. Always re-baseline vanilla on whatever runner you're using before drawing conclusions.

**2. The CIFAR 100ep proxy was useless for discriminating variants.**
- All R2 variants at 100ep landed within seed noise of vanilla (22.01, 22.23, 22.31).
- At 150ep they separate cleanly (12.5 vs 14.2 vs 21.7).
- Cause: 100ep is still in the steep descent phase. Don't pilot below 150ep on CIFAR.

**3. v02's λ-sweep is FLAT across 20× range (0.1→2.0 → FID 12.50→12.91).**
- The mining + cosine-contrastive *shape* is what helps, not the weight.
- Don't burn more compute tuning λ; the design point is structural.

**4. v01's saturating hinge was provably broken** (`neg=0` every epoch even at margin=50 on extension run). Cosine contrastive (v02) does not saturate and is strictly better.

**5. Annealing the aux-loss weight does NOT help.** v07_anneal_v02 (λ→0 over 150ep) lands at 13.00±0.63 — tied with fixed-λ v02 (12.96±0.70). The R4 1-seed v07=12.28 was lucky variance. Curriculum hypothesis is dead.

**6. Combining v02 + v06 mechanisms does NOT help.** v08 lands at 12.90 (n=1) ≈ v02 alone. Recovery branch is redundant once contrastive mining is present. Mechanisms overlap; don't stack.

**7. v04_ebm_contrastive is degenerate** (FID 436 at 100ep): InfoNCE-on-velocity-energy without a replay buffer fights the FM target. Don't revisit without Du-2021-style replay.

**Why:** these 5 facts cost ~2 weeks of compute to learn. Re-deriving them later means re-running the same null/broken experiments.

**How to apply:**
- Before any new CIFAR variant: pilot at 150ep, not 100ep.
- Before any cross-run FID comparison: confirm both numbers came from the same harness (`projects/diff-EqM/experiments/run_variant.py`).
- Don't sweep λ on v02-family variants; treat the design as locked at λ=1.0.
- The next strategic question is *transfer to Stage B*, not more CIFAR variants.
