# Training EBMs with Diffusion Contrastive Divergences (DCD)

**Citation**: Luo, Jiang, Hu, Sun, Li, Zhang. arxiv 2307.01668 (Jul 2023). Peking U + Huawei + HIT.
**Read date**: 2026-05-19
**Read by**: claude
**Read depth**: abstract (full PDF returned 404; HTML fetch deferred)

## Setting
- Task: train classical EBMs more efficiently than SGLD-based CD.
- Model: classical EBM (energy function E(x)).
- Datasets: synthetic, image denoising, CelebA-32 generation.

## Method
Replaces SGLD in CD with "EBM-parameter-free diffusion processes". Negative samples generated via forward diffusion process instead of MCMC on the EBM gradient. Frames CD as a special instance of the DCD family.

## What this paper supports
- Diffusion-based negative sampling for EBM training is more efficient than SGLD.
- Negatives generated via fixed forward diffusion process (data-independent).

## What this paper does NOT support
- No regression-target generative model (flow matching, EqM).
- No PGD or adversarial mining of negatives.
- No claim about scaling to ImageNet-256 — only CelebA-32.

## Relevance to v10 / diff-EqM
**Threat level**: LOW (different setting).

**Differentiation from us**: DCD samples negatives via fixed forward diffusion process for classical EBM training. v10 mines hard examples via PGD on the EqM regression loss. Different sampling mechanism, different model family.

**Action**: cite briefly as background for negative-mining in EBMs. Not a primary related work.

## Open questions / takeaways
1. Need to re-fetch HTML body — 404 on first attempt. Retry in Round 2.
2. Does DCD's "noised negatives at intermediate diffusion timesteps" idea overlap with our retired v03 variant? v03 was eliminated (FID 22.01 / 14.10), so this prior may not help anyway.

## Reference links
- arxiv: https://arxiv.org/abs/2307.01668
- HTML: https://arxiv.org/html/2307.01668 (returned 404 — re-fetch)
