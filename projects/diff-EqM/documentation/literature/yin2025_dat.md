# DAT: Joint Discriminative-Generative Modeling via Dual Adversarial Training

**Citation**: Yin, Zhang, Steele, Shavit, Wang. *Scalable EBMs via Adversarial Training (DAT)*. arxiv 2510.13872 (Oct 2025). MIT + Independent.
**Read date**: 2026-05-19
**Read by**: claude
**Read depth**: full method + tables

## Setting
- Task: hybrid discriminative + generative; classifier-as-EBM.
- Model family: classical EBM (JEM-style).
- Datasets: CIFAR-10/100, ImageNet-256.
- Best FID: ImageNet-256 WRN-50-4 T=65 **FID 5.39, IS 341.9**; ResNet-50 T=30 FID 5.88.

## Method
BCE discriminative loss + PGD-generated negatives:

```
L_BCE = −E_x~p_data [log σ(−E(x))] − E_x~p_θ [log(1 − σ(−E(x)))]
```

Generative PGD: T=15-70 unconstrained, step η=2.0 on IN, η=0.1 on CIFAR. Plus discriminative L_AT-CE with L2-bounded PGD (ε=3.0 IN, 0.5 CIFAR).

Two-stage training: Stage 1 BN-on discriminative; Stage 2 BN-off generative.

## What this paper supports
- PGD-generated negatives can train scalable EBMs at IN-256 (first hybrid at this scale).
- BCE discrimination + adversarial robustness loss compose.
- Beats BigGAN-deep (6.95).

## What this paper does NOT support
- NO regression-target / velocity-field formulation.
- NO equilibrium matching, flow matching, or implicit-gradient-field model.
- NO coverage of post-training of pretrained generative models.
- Sampling = PGD on energy (T=15-70 steps per sample) — expensive at inference.

## Relevance to v10 / diff-EqM
**Threat level**: HIGH-prior, MED-scoop.

**Differentiation from us**:
- DAT: PGD generates **discriminative negatives**, used in BCE loss; classifier-as-EBM.
- v10+CAFM: PGD generates **hard examples for regression loss**; GAN-style discriminator separately; applied to EqM (implicit gradient field, NOT classifier).

**Why this is not a scoop**: DAT's PGD serves discrimination, not regression. v10's PGD targets the EqM L2 regression objective. Mechanically distinct.

**Action**: cite as nearest PGD-EBM prior. Explicitly contrast: "DAT applies PGD to generate negatives for discriminative BCE training in classical EBMs; v10 applies PGD to generate hard examples for the L2 regression training of implicit-energy models."

## Open questions
1. Could DAT's two-stage training (BN-on/off) inform our combined v10+CAFM schedule? Likely yes for stability.
2. DAT's IN-256 5.39 FID with T=65 sampling steps. Lin CAFM 1.53 FID with vanilla sampling. v10+CAFM target: beat CAFM. DAT is not the bar.

## Reference links
- arxiv: https://arxiv.org/abs/2510.13872
- HTML v1: https://arxiv.org/html/2510.13872v1
