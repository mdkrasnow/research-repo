# ACT-Diffusion: Efficient Adversarial Consistency Training for One-step Diffusion Models

**Citation**: Kong et al. *ACT-Diffusion*. CVPR 2024. arxiv 2311.14097.
**Read date**: 2026-05-19
**Read by**: claude
**Read depth**: method

## Setting
- Task: one-step diffusion generation via improved consistency training.
- Model family: consistency models (Song 2023) + discriminator.
- Datasets: CIFAR-10, ImageNet 64, LSUN Cat 256.
- Best FID: not extracted in our fetch (specific numbers; abstract claims "improved FID").

## Method
Combines consistency loss with discriminator-based adversarial term.

```
L_CT = Σ_k E[d(f(x₀ + t_k z, t_k), f(x₀ + t_{k-1} z, t_{k-1}))]   # consistency
L_G = log(1 − D(f(x + t_{n+1} z, t_{n+1}), t_{n+1}))               # adversarial
L_f = (1 − λ_N(k)(n+1)) · L_CT + λ_N(k)(n+1) · L_G                  # combined
```

Schedule λ_N increases adversarial weight at later timesteps. Discriminator is time-conditioned: `D(x_t, t)`.

## Key technical choices
- StyleGAN2-style gradient penalty + lazy regularization.
- Time-conditioned discriminator outperforms time-unconditioned empirically.
- λ schedule: increases monotonically with consistency-error accumulation.
- Less than 1/6 baseline batch size, 1/2 parameters/steps.

## What this paper supports
- Combining discriminator AT with regression-based consistency loss works.
- Time-conditioning the discriminator helps.
- Adversarial term can compensate for accumulated consistency error.

## What this paper does NOT support
- **NO input-space adversarial perturbation (PGD or FGSM)** anywhere.
- No ablation testing PGD-on-input alongside the discriminator.
- No coverage of EqM, flow matching, or implicit-energy regression.

## Relevance to v10 / diff-EqM
**Threat level**: MEDIUM (adjacent prior; doesn't subsume v10+CAFM).

**Differentiation from us**:
- ACT: discriminator + consistency loss; no PGD-input mining.
- v10+CAFM: discriminator + L2-regression-on-perturbed-input (PGD mined); applied to EqM/SiT.

**Why this doesn't kill Branch B**: ACT proves the "discriminator + regression loss" composition idea on consistency models. We extend it by adding a third orthogonal signal: PGD hard-example mining. ACT can be cited as proof-of-concept for the composition; v10 is the additional novelty.

**Action**: 
- Cite as prior work demonstrating discriminator+regression composition.
- Use ACT's λ schedule as design reference for Phase 2 hyperparameter tuning.
- Explicitly state v10's PGD-on-input is what ACT does NOT do.

## Open questions
1. Could v10+CAFM be reframed as "PGD extension of ACT"? Useful if reviewers question ACT differentiation.
2. ACT's "time-conditioned discriminator" decision validates CAFM's approach. Consistent design choice across the field.

## Reference links
- arxiv: https://arxiv.org/abs/2311.14097
- HTML v2: https://arxiv.org/html/2311.14097v2
