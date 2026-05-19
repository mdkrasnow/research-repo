# What is Adversarial Training for Diffusion Models?

**Citation**: Briglia, Mirza, Lisanti, Masi. *What is Adversarial Training for Diffusion Models?* arxiv 2505.21742 (May 2025).
**Read date**: 2026-05-19
**Read by**: claude
**Read depth**: method + tables + critical analysis

## Setting
- Task: train robust diffusion models for image generation under corrupted data.
- Model: DDPM/DDIM.
- Datasets: CIFAR-10, CelebA, LSUN Bedroom (all with corruption rates p ∈ {0%, 50%, 90%}).
- Best result: CIFAR-10 90%-corrupted FID 24.81 (vs DDPM 102.68).

## Method
**Central argument**: AT for classifiers enforces **invariance** (same output under perturbation). AT for diffusion must enforce **equivariance** (output shifts proportionally to input shift).

Naive invariance loss:
```
L_invariance = ||ε_θ(x_t + δ, t) − ε_θ(x_t, t)||²
```
This **fails catastrophically** — Figure 3 shows FID > 200 at any reasonable λ.

Their proposed equivariance loss (Eq. 9):
```
L_AT = ||ε_θ(x_t, t) − ε||² + λ_t · ||ε_θ(x_t + δ, t) − [ε_θ(x_t, t) + δ]||²
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^
                                                    equivariance: output shifts by δ too
λ_t = (λ · √3) / (β · r(t))
```

Adversarial perturbation: **single-step FGSM with random start**, NOT multi-step PGD. Perturbation magnitude r(t) follows noise scheduler σ(t).

## Key technical choices
- Single-step FGSM (no PGD). λ=0.3 recommended.
- Training overhead ×2.5 vs vanilla DDPM (one extra forward+backward).
- Tested on corruption (training data noisy), not on clean-data generation gain.

## What this paper supports
- Adversarial regularization works for **robust denoising** under corrupted training data.
- Equivariance formulation is correct framing for diffusion AT (output shifts with input).
- Single-step FGSM sufficient; iterative PGD unnecessary in their setting.
- λ should scale inversely with timestep noise level.

## What this paper does NOT support
- Does NOT show AT improves FID on **clean-data** generation (FID slightly worse on p=0% per Table 2: CIFAR-10 Robust_adv 28.68 vs baseline 7.2).
- Does NOT test multi-step PGD; provides no direct evidence against it but suggests overhead concern.
- Does NOT test L2-regression-on-FIXED-target at perturbed input (= v10 formulation).
- Does NOT cover flow matching, EqM, or regression-target implicit-energy models.

## Relevance to v10 / diff-EqM
**Threat level**: **HIGH (mechanism threat)**.

**Differentiation from us**: 
- Briglia: equivariance loss with FGSM single-step. Their loss = `||f(x+δ) − [f(x) + δ]||²` (output should track input perturbation).
- v10: `||f(x+δ*) − target(x,ε,γ)||²` where target is the SAME as clean. Output should hit the SAME target despite perturbation.

**Critical question**: is v10's formulation invariance-leaning?

**Analysis**:
- v10 asks: at any nearby `x_γ + δ`, predict the SAME `(ε−x)·c(γ)` target as at clean `x_γ`.
- This is NOT the v01-style "minimize ||f(x+δ) − f(x)||²" pure invariance.
- v10 anchors to the GROUND-TRUTH regression target, not to the clean prediction. The model is allowed to behave differently at perturbed points — it just must satisfy the true target there.
- **v10 is closer to standard regression-with-augmentation than to invariance.** It says "the regression task should still hold under adversarial perturbation."
- Equivalent framing: v10 is robust regression. Briglia's framing requires the OUTPUT to be equivariant; v10 requires the TARGET-SATISFACTION to be invariant under input perturbation.
- These are NOT identical concerns. v10 may be safe.

**However**, Briglia's argument deserves more thought. There may be a mode where:
- At `x_γ + δ`, the true EqM target is well-defined (we know `(ε−x)·c(γ)` exactly because (x,ε) generated x_γ).
- But the **same target** evaluated at the **perturbed input** doesn't correspond to the same (x,ε) — it would correspond to a different (x',ε') such that `γ·x' + (1−γ)·ε' = x_γ + δ`.
- So v10 is asking the model to ignore the perturbation and predict as if it weren't there. **This IS a form of invariance**.

**Decision**: v10 mechanism is plausible but the equivariance framing matters. Two follow-ups:
1. Read Briglia full paper in Phase 0 — confirm whether they tested anything like v10's formulation in ablations.
2. Add a "v10-equivariant" variant to fallback list: `L_aux = ||f(x+δ) − f(x).detach() − Jδ||²` where Jδ is linearized correction. May be safer alternative.

**Action**: 
- Cite as mechanism-design reference in v10 design doc.
- Add equivariant-v10 variant to v11 fallback sketch.
- Test small-λ v10 first; if collapse, switch to equivariant formulation.

## Open questions / takeaways
1. v10 stability under aggressive λ — Briglia shows invariance-flavored losses collapse at strong λ. Phase 0 CIFAR sanity must include diagnostic for this.
2. Single-step FGSM may be sufficient — could reduce v10 K=3 to K=1, save compute.
3. Their phase-dependent λ_t = c · σ(t)⁻¹ scaling suggests v10 λ should be γ-dependent. Possible HP improvement.

## Reference links
- arxiv: https://arxiv.org/abs/2505.21742
- HTML: https://arxiv.org/html/2505.21742v1
