# Workshop Paper Draft — §2 Background

Status: v1 draft, 2026-05-20.

---

## 2. Background and Related Work

We position our contribution at the intersection of three recent threads: regression-target generative models (flow matching, EqM), adversarial training of generative models (CAFM, DAT, ACT, AEBM-Diff), and contrastive supervision on velocity fields (VeCoR). We summarize each, then identify the specific gap closed by our work.

### 2.1 Regression-target generative models

A growing class of generative models replaces the maximum-likelihood or score-matching objective with a direct ℓ₂ regression on a deterministic target derived from a coupling between noise and data. Given samples ``x ∼ p_data`` and ``ε ∼ N(0, I)``, the network ``f_θ`` is trained to predict a velocity-like quantity at the interpolated point ``x_t = (1−t)x + tε``:

```
L(θ) = E_{x,ε,t} ‖ f_θ(x_t, t) − u(x, ε, t) ‖²
```

where ``u`` is the target field. Different choices of ``u`` give different generative paradigms:

- **Flow matching (Lipman et al. 2023)** sets ``u = ε − x``, the velocity of a straight conditional probability path. Inference integrates the ODE ``dx/dt = −f_θ(x_t, t)``. Generalizations include rectified flow (Liu et al. 2022), interpolant frameworks (Albergo et al. 2023), and Scalable Interpolant Transformers (SiT, Ma et al. 2024), which adapts the DiT backbone (Peebles & Xie 2022) to the interpolant framework and achieves FID 2.06 on ImageNet-256.

- **Consistency models (Song et al. 2023)** regress on a self-consistency target across diffusion timesteps to enable one-step generation. Variants include improved consistency training (Song & Dhariwal 2023) and stable consistency tuning (2024).

- **Equilibrium Matching (Wang & Du 2025)** removes time conditioning entirely: ``f_θ(x_γ)`` predicts an energy-compatible gradient field ``(ε − x) · c(γ)``, where ``c(γ)`` is a piecewise-linear schedule with ``c(1) = 0``. The model implicitly defines an energy landscape ``E`` such that ``f_θ ≈ ∇E``, and sampling reduces to gradient descent on ``E`` with a NAG-style optimizer. EqM achieves FID 1.90 on ImageNet-256 with an XL/2 model at 1400 epochs of training, surpassing the equivalent SiT-XL/2 (FM, FID 2.06). Wang & Du note that "sensitivity to adversarial perturbations, spurious minima, dataset biases, and privacy risks in gradient-based samplers is unexamined in the current work."

All three paradigms share the same fundamental signal: a deterministic regression target derived from a noise-data coupling, optimized with mean-squared error. Each sample contributes equally to the loss regardless of how well the model already fits it.

### 2.2 Adversarial training for generative models

A separate research thread augments these regression objectives with an adversarial component. Recent work falls into three distinct mechanisms:

**Discriminator-based adversarial post-training.** Lin et al. (2026) introduce Adversarial Flow Models (AFM) and Continuous Adversarial Flow Models (CAFM). CAFM replaces flow matching's fixed ℓ₂ criterion with a least-squares GAN loss in which a learned discriminator ``D_φ`` operates on velocity-field derivatives via Jacobian-vector products (JVP). The training schedule alternates 16 discriminator updates per generator update over 10 epochs of post-training on top of a 1400-epoch FM baseline. On SiT-XL/2 ImageNet-256, this improves guidance-free FID from 8.26 to **3.63** and classifier-free guided FID from 2.06 to **1.53** — state-of-the-art among flow-matching variants. The discriminator's signal is global: it scores the marginal distribution of predicted velocities against the target distribution at each timestep. We use CAFM as a starting point for our combination.

**Discriminative adversarial training for classifier-as-EBM.** Wu et al. (2025, DAT) interpret a classifier as an energy-based model (following JEM, Grathwohl et al. 2020) and replace its SGLD-based MCMC negative sampling with PGD-generated contrastive negatives in a binary-cross-entropy loss. On ImageNet-256, DAT achieves FID 5.39 with a WRN-50-4 backbone — the first classical EBM-hybrid scaled to this resolution. DAT's PGD operates on a discriminative loss surface (BCE on energy values), not on a regression target, so the negative-mining objective is fundamentally different from ours.

**Generator-discriminator adversarial EBMs over diffusion.** Geng et al. (2024, AEBM-Diff) embed an EBM at each denoising timestep and train via a generator-discriminator loop with symmetric Jeffrey divergence. This is GAN-style adversarial training, not PGD-based hard-example mining.

**Discriminator-augmented consistency training.** Kong et al. (2024, ACT) add a discriminator to consistency training, weighted by a schedule that grows the adversarial term at later timesteps. ACT demonstrates that combining a discriminator with a regression-based generative objective is feasible and improves FID on CIFAR-10, ImageNet-64, and LSUN-Cat-256. Notably, ACT does **not** perform any input-space adversarial perturbation alongside the discriminator.

**Older PGD+GAN combinations.** Rob-GAN (Liu & Hsieh 2019) couples a PGD attacker with a GAN discriminator for jointly improving generator quality and discriminator robustness, but targets classification-style real-vs-fake discrimination on full images, not regression-target adversarial mining on a velocity field.

### 2.3 Contrastive supervision on velocity

VeCoR (Hong et al. 2025) introduces a two-sided contrastive loss on the velocity field, attracting predictions toward a ground-truth positive and repelling them from heuristic negatives generated via data-agnostic augmentations (random crop, channel shuffle, CutMix, additive noise). On SiT-XL/2 ImageNet-256, VeCoR yields **22% relative FID reduction** (20.01 → 15.56), and on REPA-SiT-XL/2 the gain is **35% relative** (11.14 → 7.28). In their concluding section, the authors explicitly note that "the current negative sampling strategy remains heuristic and data-agnostic" and identify **adaptive hard-negative mining** and **trajectory-aware perturbations** as open future work.

### 2.4 Adversarial training of diffusion models: invariance vs equivariance

Briglia et al. (2025) provide a theoretical analysis of what adversarial training should mean for a regression-target generative model. They argue that the standard adversarial training prescription for classifiers — output **invariance** under input perturbation — fails for diffusion: enforcing ``‖f(x+δ) − f(x)‖²`` causes catastrophic FID degradation (Figure 3 of their paper, reaching FID > 200 at non-trivial regularization strength). Instead they prescribe **equivariance**: the output should shift proportionally to the input perturbation, ``f(x+δ) ≈ f(x) + δ``, which preserves the diffusion trajectory's coupling to the data manifold. They experimentally use single-step FGSM rather than multi-step PGD, motivated by training-cost considerations.

Briglia's analysis is the principal mechanism-level threat to any invariance-flavored adversarial training of regression-target models. We discuss in §3.2 why our v10 objective — anchored to a fixed ground-truth target rather than the clean network output — is robust to this failure mode, and report in §4.1 that v10 trains stably for 150 epochs on CIFAR-10 without the collapse signature Briglia identifies.

### 2.5 Hard-example mining: classical context

Outside generative modeling, hard-example mining has a long history in supervised learning. Online Hard Example Mining (OHEM, Shrivastava et al. 2016) selects the highest-loss samples from each mini-batch for backward pass; Focal Loss (Lin et al. 2017) implicitly down-weights easy examples via a confidence-modulated factor; contrastive learning (InfoNCE, SimCLR) requires explicit hard-negative construction for representation quality. The PGD adversarial training framework of Madry et al. (2017) generalizes this to an inner-maximization problem over a bounded perturbation set, which our v10 inherits structurally while replacing the cross-entropy outer loss with the EqM regression loss.

### 2.6 Position of our work

Our contribution is the **specific combination** that none of the above papers explores:

- Like VeCoR, we add a per-sample loss term targeting hard negatives — but our negatives are **adversarially mined against the regression loss** (per VeCoR's own future-work direction), not drawn from fixed augmentations.
- Like CAFM, we add a discriminator-based adversarial term — but we apply it for the first time to the EqM family, and we combine it with the v10 mining term to test whether discriminator-based and PGD-based adversarial signals are complementary.
- Like Madry and Briglia, we use PGD-based adversarial mining — but we target a regression objective in a generative model rather than classifier output invariance, and we use the equivariance-preserving formulation (anchoring to ground-truth target) that Briglia's analysis would predict to be stable.
- Like DAT, we use PGD-generated negatives in adversarial training — but our negatives feed a regression loss rather than a discriminative BCE loss, and our target model is an implicit-energy regression model rather than a classifier-as-EBM.
- Like Rob-GAN, we combine PGD with discriminator-based adversarial training — but our PGD targets the regression loss surface of the generator (not the discriminator's loss surface), and our generative model is a regression-target implicit-energy field (not an image-level GAN).

To our knowledge, this combination — **PGD hard-example mining on a regression target, combined with discriminator-based adversarial post-training, applied to an implicit-energy generative model** — is novel.

---

## Notes for revision

- §2.6 may be cut for length; the differentiation is implicit in §2.1-2.5 and could be summarized in a single sentence with a comparison table relegated to appendix.
- Need to add: Du & Mordatch 2019 (classical EBM-CD) as lineage citation for EqM/Du; FlowEqProp (Gower 2026) clarifying it's unrelated to EqM-Wang.
- Consider a single Related Work table summarizing axes: (model family, loss type, adversarial mechanism, mining mechanism, datasets) across {ours, CAFM, DAT, ACT, AEBM-Diff, VeCoR, Rob-GAN, Briglia}.
- Once Phase 5 SiT head-to-head numbers exist, §2.2 paragraph on CAFM should explicitly mention our reproduction.
