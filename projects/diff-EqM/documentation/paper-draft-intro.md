# Workshop Paper Draft — Introduction Section

Status: draft v1, 2026-05-19. To be iterated when CIFAR sanity + Phase 1b numbers arrive.

---

## Abstract (draft, ≤200 words)

Recent work has scaled adversarial training to regression-target generative models. Lin et al. (2026, CAFM) introduce a learned discriminator that operates on velocity fields, achieving SOTA FID on ImageNet-256 SiT by replacing flow matching's plain MSE criterion with adversarial supervision. In parallel, VeCoR (Hong et al., 2025) demonstrates that contrastive supervision on negative velocity samples yields large FID gains, and explicitly identifies adaptive hard-negative mining as open future work. We close this gap: we introduce v10, a PGD-based hard-example mining objective on the velocity-field regression target itself, and show that it composes with CAFM-style discriminator post-training to compound FID gains beyond either alone. Applied to Equilibrium Matching (Wang & Du, 2025) — a regression-based implicit-energy model — our combined v10+CAFM achieves [X] FID at EqM-B/2 ImageNet-256, improving over both vanilla EqM (31.41) and CAFM-only (Y) baselines across 3 seeds. The two losses are mechanistically complementary: the discriminator catches global distributional mismatch while PGD-mining catches local regression failure. We also demonstrate (in concurrent ICLR full paper) head-to-head transfer on SiT, confirming the result generalizes beyond EqM.

## 1. Introduction

Regression-target generative models — flow matching (Lipman et al., 2023), Scalable Interpolant Transformers (Ma et al., 2024), and Equilibrium Matching (Wang & Du, 2025) — currently achieve state-of-the-art FID on ImageNet-256 by regressing on velocity-field targets. Training is plain ℓ₂: the network ``f(x_t, t)`` predicts the velocity (or, in Equilibrium Matching, a c(γ)-scaled gradient field), and the loss is mean-squared error against a deterministic target. The simplicity is attractive, but it leaves a fundamental signal on the table: every sample contributes equally regardless of difficulty.

Two recent works have begun to address this:

**Discriminator-based adversarial post-training.** Lin et al. (2026) introduce Continuous Adversarial Flow Models (CAFM): they post-train a pretrained flow-matching model by replacing the fixed ℓ₂ loss with a learned discriminator that operates on the velocity field via Jacobian-vector products. On SiT-XL/2, this improves guidance-free ImageNet-256 FID from 8.26 to 3.63 — a 56% relative improvement over the regression baseline. The discriminator provides a global distributional signal: it knows what real velocity fields look like, in aggregate, at each timestep.

**Contrastive supervision on velocity.** Hong et al. (2025, VeCoR) propose a two-sided contrastive loss on the velocity field with negatives drawn from data-agnostic augmentations (channel shuffle, CutMix, additive noise). On SiT-XL/2, VeCoR yields 22% relative FID reduction. Crucially, the authors explicitly note in their conclusion that "the current negative sampling strategy remains heuristic and data-agnostic" and identify **adaptive hard-negative mining** as open future work.

**The gap.** Neither line of work performs adversarial mining of inputs against the regression objective itself. CAFM's adversarial signal lives in the discriminator network; VeCoR's negatives are augmentation-based and fixed. The natural question — "can we use the model's own regression error to mine hard examples, and does this compose with discriminator-based supervision?" — has not been studied for regression-target generative models. (For *discriminative* energy-based models, Wu et al. 2025 (DAT) explore PGD-generated negatives with a BCE classifier loss; for adversarial-EBM-diffusion, Geng et al. 2024 use generator-discriminator GAN dynamics. Both target different model families and different loss types than the regression-target setting we address.)

**This work.** We introduce **v10**, a PGD hard-example mining objective on the velocity-field regression target:

```
δ* = argmax_{||δ||₂ ≤ ε}  || f(x_t + δ) − target ||²
L_v10 = ||f(x_t + δ*) − target||²
```

with default K=1 (FGSM-style; Briglia et al. 2025 show single-step is sufficient and stable for diffusion AT), ε=0.3, and small λ=0.1. We then combine v10 with CAFM's discriminator-based adversarial post-training:

```
L_G = L_CAFM_generator + λ_v10 · L_v10
```

trained with Lin's N=16 discriminator-updates-per-generator-update schedule and 10-epoch post-training recipe.

We make two contributions:

1. **First port of CAFM-style adversarial post-training to Equilibrium Matching.** EqM is time-unconditional (the network sees t=0); we adapt CAFM's time-conditioned JVP discriminator to use the mixing coefficient γ in place of t, and replace the flow-matching velocity target ε − x with EqM's c(γ)-scaled gradient target (ε − x)·c(γ). On vanilla EqM-B/2 80ep (FID 31.41), CAFM-only post-training achieves FID [Y].

2. **First combination of PGD-input-mining with discriminator-based adversarial training on regression-target generative models.** Across three seeds on EqM-B/2 ImageNet-256, v10+CAFM achieves FID [X], improving over both CAFM-only ([Y]) and v10-only ([Z]). Diagnostics confirm the two losses are non-redundant: L_v10 (regression error at adversarial inputs) and L_disc (discriminator's score on generated vs real velocities) both decrease during training but at different rates and with different sensitivity to model capacity, indicating they constrain different failure modes.

We also report (extended results, ICLR full paper) head-to-head experiments on SiT-XL/2 using Lin's publicly released pretrained checkpoint, confirming the combination generalizes beyond EqM to standard flow matching.

**Why this matters.** Adaptive hard-negative mining was identified as open future work by Hong et al. (2025). Our work resolves that direction in the specific setting of regression-target adversarial training: yes, adaptive mining works, and yes, it composes with the dominant prior approach (discriminator AT). The mechanism is simple, plug-in (one additional forward+backward per generator step), preserves the regression formulation (no extra inference cost), and uses only the model's own loss surface (no architectural changes).

## 2. Background

[…filled in §3.5 of `documentation/workshop-paper-outline.md`…]
