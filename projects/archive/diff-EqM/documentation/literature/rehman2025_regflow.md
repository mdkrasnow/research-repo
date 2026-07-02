# RegFlow: Efficient Regression-Based Training of Normalizing Flows for Boltzmann Generators

**Citation**: Rehman, Davis, Lu, Tang, Bronstein, Bengio, Tong, Bose. *RegFlow*. ICML GenBio Best Paper 2025. arxiv 2506.01158.

## Setting
- Task: Boltzmann generators for molecular conformations (alanine peptides).
- Model: normalizing flows trained via regression.

## Method
Replaces maximum-likelihood training with **ℓ₂-regression on samples from OT plans or pretrained CNFs**. Adds forward-backward self-consistency loss.

## Relevance
**Threat level**: LOW (different domain — molecular Boltzmann, not image generation).
**Differentiation from us**: scientific application, no PGD, no discriminator AT, no image FID benchmark.
**Useful cite**: support for "regression-based training of flow models is gaining traction" in our introduction.

## Action
- Optional cite in introduction as evidence that regression-target flow models are an active research direction beyond image gen.
- No scoop threat.
