# Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models

**Citation**: Wang, R. & Du, Y. *Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models*. arxiv 2510.02300 (Oct 2025).
**Read date**: 2026-05-19
**Read by**: claude
**Read depth**: method + tables

## Setting
- Task: class-conditional image generation IN-256.
- Model family: implicit energy-based (gradient field of implicit landscape).
- Datasets: ImageNet 256, CIFAR-10.
- Best FID: **1.90** (EqM-XL/2, 1400ep, NAG-GD sampler, 250 NFE).

## Method
Network f outputs a gradient field. Trained to predict `target = (ε − x) · c(γ)` where:
- `ε ~ N(0,I)`, `x_γ = γ·x + (1−γ)·ε`, γ ~ U(0,1)
- c(γ) truncated decay: c=1 for γ≤a, c=(1−γ)/(1−a) for γ>a; default **a=0.8**
- Global gradient multiplier **λ=4.0** on ImageNet
- Loss: `L = ||f(x_γ) − (ε−x)·c(γ)||²`

f implicitly represents `∇E`. Statement 1: at perfect training, `||f(x_data)|| ≈ 0`. Statement 2: local minima of E ≈ data points. Convergence O(1/K) under L-smooth E.

## Key technical choices
- Architecture: **SiT (Ma 2024) backbone exactly**. No architectural changes. Patch 2×2. Sizes S/B/L/XL.
- **Time conditioning removed (input t=0)** — this is the equilibrium property.
- Optimizer: AdamW, LR 1e-4 constant, batch 256.
- Sampler: NAG-GD η=0.0017, μ=0.3, 250 steps default.
- Explicit-energy variants (EqM-E) tested but worse — implicit gradient field preferred.

## What this paper supports
- Implicit energy via gradient field can match flow-matching / diffusion FID at scale.
- Sampling via gradient descent on landscape works with adaptive compute.
- Beats SiT-XL/2 (FM, 2.06), DiT-XL/2 (2.27), VDM++ (2.12), StyleGAN-XL (2.30) at XL/2 1400ep.
- **EqM-B/2 80ep ~ 2.8 FID** (Figure 3 scaling; class-conditional IN-256).

## What this paper does NOT support
- No adversarial training, no PGD, no discriminator.
- No claim about robustness to input perturbation.
- CIFAR underperforms vanilla Flow Matching (Table 8): "over-optimization on the CIFAR-10 baseline."
- No classifier-free guidance results — all numbers unconditional class-conditional.

## Relevance to v10 / diff-EqM
**Threat level**: INFO (base model, not a competitor). HIGH SECONDARY because Yilun Du is a senior author and Du's prior work covers adversarial EBM-CD training — possible parallel adversarial-EqM work risk.

**Differentiation from us**: EqM is the base model we extend. v10 + CAFM both add adversarial training; Wang+Du baseline has neither.

**Action**: cite as base model. Watch Yilun Du for follow-ups.

## Open questions / takeaways
1. **CRITICAL: baseline reconciliation.** Paper Figure 3 shows EqM-B/2 80ep ~2.8 FID class-conditional IN-256. Our trusted baseline `stage_b_vanilla_in1k_80ep_seed0` reports FID 31.41 (paper-comparison Δ=−1.44 vs paper 32.85). The 32.85 vs 2.8 discrepancy is enormous. **Likely explanations**:
   - Paper Fig 3 uses XL/2 only, B/2 not actually 2.8 — need re-read.
   - Possible config mismatch: classifier-free guidance, EMA, sampler tuning.
   - Our 31.41 may be guidance-free; paper number includes CFG.
   - **MUST resolve before Phase 1.** Investigate in Phase 0.1 alongside v02 postmortem.
2. EqM removes time conditioning — does CAFM's JVP discriminator (which takes (x_t, t)) need adaptation for EqM's t=0? **Yes.** Plan port-design accordingly.
3. Sampler is gradient descent on landscape, not ODE integration. Our diagnostics + FID pipeline must support this.

## Reference links
- arxiv: https://arxiv.org/abs/2510.02300
- HTML v3: https://arxiv.org/html/2510.02300v3
- code: github.com/raywang4/EqM
- project page: https://raywang4.github.io/equilibrium_matching/
