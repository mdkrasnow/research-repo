# VeCoR: Velocity Contrastive Regularization for Flow Matching

**Citation**: Hong, Li, Li, Zhang, Tang (JIIOV Technology). *VeCoR*. arxiv 2511.18942 (Nov 2025).
**Read date**: 2026-05-19
**Read by**: claude
**Read depth**: method

## Setting
- Task: flow matching for image generation.
- Model: SiT-XL/2, REPA-SiT-XL/2.
- Datasets: ImageNet-256 (50 NFE Euler-Maruyama).
- Best FID: REPA-SiT-XL/2 11.14 → 7.28 (35% relative).

## Method
Auxiliary contrastive loss on velocity field:

```
L_VeCoR = (1/N) Σᵢ [||v_θ(x̂_t, t) − v̂₊||² − λ Σⱼ₌₁ᴷ ||v_θ(x̂_t, t) − v̂₋ⱼ||²]
```

- v̂₊ = ground-truth velocity (positive).
- v̂₋ = perturbed velocity (negative).
- L2 distance, not cosine.
- Negatives via **augmentation-style perturbations**: image-space (random crop, channel shuffle, CutMix, blur, color jitter), latent-space same, or velocity-space (direct perturbation).
- Default: K=1, channel shuffle in velocity space.
- Stability: λK < 1.

## Key technical choices
- Heuristic, data-agnostic perturbations. NOT adversarial mining.
- Tested on SiT-XL/2 and REPA-SiT-XL/2 at 50 NFE.
- Reports SiT-XL/2 baseline FID 20.01 → 15.56 (22%). REPA-SiT-XL/2 11.14 → 7.28 (35%).

## What this paper supports
- Two-sided contrastive supervision (attract positive, repel negative) improves flow-matching FID.
- Augmentation-based negatives suffice for measurable gain.
- Per-step overhead modest.

## What this paper does NOT support
- **NO adversarial mining of negatives.** All perturbations are predefined augmentations.
- No PGD, no FGSM, no L2-regression-on-target adversarial training.
- No ablation comparing adversarial vs augmentation-based negatives.
- No coverage of EqM family (only SiT and REPA-SiT).

**Authors' own future work statement (CRITICAL)**: "the current negative sampling strategy remains heuristic and data-agnostic" and identifies "**adaptive hard-negative mining and trajectory-aware perturbations**" as open future work.

## Relevance to v10 / diff-EqM
**Threat level**: MEDIUM, but **CITATION GOLD for v10 motivation**.

**Differentiation from us**:
- VeCoR: predefined augmentation-style perturbations.
- v10: adversarial PGD mining on the same regression objective.
- v10+CAFM: above + discriminator AT.

**Why this is strong support for v10**: VeCoR explicitly cites "adaptive hard-negative mining" as open future work. v10 is exactly that. Citing this future-work statement in our paper directly justifies the v10 direction.

**Action**:
- Cite VeCoR as motivating prior for the hard-mining direction.
- Use their future-work statement verbatim in introduction.
- VeCoR's 22-35% relative FID gain on SiT-XL/2 is the bar for v10 + CAFM combined gains.
- Their λK < 1 stability constraint may apply to v10 too — check empirically.

## Open questions
1. Do v10's PGD-mined negatives produce more aggressive gradient signal than VeCoR's augmentation? Compare diagnostics.
2. VeCoR works on SiT (time-conditioned). Does augmentation-based VeCoR transfer to EqM (time-unconditional, t=0)? If yes, becomes another baseline to beat.

## Reference links
- arxiv: https://arxiv.org/abs/2511.18942
- HTML v1: https://arxiv.org/html/2511.18942v1
