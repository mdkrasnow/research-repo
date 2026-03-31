# IRED-CD Contrastive Investigation — Final Analysis
**Date**: 2026-02-22
**Status**: COMPLETE — Investigation closed

---

## Executive Summary

The IRED-CD contrastive negative mining investigation is complete. After 6 ablation experiments (q211–q216), the conclusion is clear:

**No contrastive approach on scalar EBM energies improves over the MSE-only baseline (val MSE = 0.00969). The gradient-contrastive approach (q216) matches baseline performance exactly without breaking training, but does not improve it.**

The project's best result remains: **val MSE = 0.00969 ± 0.00002** (q211/q101 baseline, no mining).

---

## Ablation Results Table

| Exp | Description | val MSE | pred_norm@100k | E_pos@100k | Verdict |
|-----|-------------|---------|----------------|------------|---------|
| **q211** | MSE only, no mining (anchor) | **0.00969** | 19.4 | 447k | ✓ Baseline |
| q212 | energy_reg=0.01 + CL | 3.828 | 0.0006 | ~1.7 | ✗ reg kills pred |
| q213 | no reg + CL (normal) | 0.89–1.66 | 0.25–0.39 | ~2,700 | ✗ CL suppresses E_pos |
| q214 | no reg + CL (detach E_pos) | 3.91 | 18,354 | ~9e13 | ✗ shared-param explosion |
| q215 | reg=1e-10 + CL (detach E_pos) | 3.645 | 0.267 | ~841k | ✗ Goldilocks doesn't exist |
| **q216** | gradient-contrastive | **0.00974** | **19.4** | N/A | ✓ Works, matches baseline |

---

## Structural Incompatibility: Energy-Value Contrastive vs IRED Parameterization

### Root cause (confirmed by q212–q215)

In this architecture, `pred = ∂E/∂x / B_norm`. For denoising to work:
- Need `||∂E/∂x|| ≈ B_norm × ||ε||₂ ≈ 2048 × 20 ≈ 40,000`
- This requires `E ≈ 447k` (natural equilibrium under MSE-only training)

Any energy-value contrastive term `L_CL = f(E_pos, E_neg)` is incompatible because:

1. **energy_reg suppresses E** (q212): reg keeps `|E| ~ 1.7`, so `||∂E/∂x|| ~ 1.2`, `pred ~ 0.0006`. Denoising dead.

2. **CL gradient suppresses E_pos** (q213): `dL_CL/dE_pos > 0` drives E_pos down. Equilibrium at E_pos ~2,700 (165× too small). pred_norm ~0.3.

3. **Detaching E_pos causes shared-param explosion** (q214): Contrastive updates θ to raise E_neg. Shared θ raises E_pos too. E_pos → 9e13. pred_norm → 18,354. MSE catastrophic.

4. **Tiny energy_reg arrests explosion but CL still flattens gradient field** (q215): E stabilizes at 841k (similar magnitude to q211), but `gradE/E ratio` = 0.00065 vs q211's 0.089 — 137× flatter. pred_norm = 0.267. The CL loss finds a regime with separable energies but minimal gradient field, avoiding the denoising cost. Val MSE = 3.645.

**Key insight**: The contrastive loss creates an alternative low-cost solution in parameter space — large, separable energies with a flat gradient field — that satisfies the contrastive objective without learning to denoise. The MSE loss cannot overcome this attraction because the flat gradient field makes its signal vanishingly small.

### Why gradient-contrastive (q216) works

`L_GC = softplus((MSE_pos.detach() - MSE_neg) / T)`

- Operates entirely in prediction space — no energy values involved
- No coupling between "contrastive objective" and "gradient field magnitude"
- E_pos grows freely under MSE supervision → reaches 447k naturally
- pred_norm = 19.4 (identical to baseline)
- val MSE = 0.00974 (2 seeds; statistically identical to baseline 0.00969)

The GC loss is **compatible** with the architecture, but provides no improvement over baseline.

---

## Why Gradient-Contrastive Does Not Improve Over Baseline

The GC loss enforces: `MSE(pred_pos, ε) < MSE(pred_neg, ε)`.

At convergence, `pred_pos ≈ ε` (MSE_pos → 0.05), and the GC loss pushes `pred_neg` away from ε (MSE_neg → 1.06). However:

- The negative samples (Langevin, noise_scale=1.5, 1 step) are at `neg_dist ≈ 14` — far from the data manifold
- The model already naturally has high error at far-from-manifold points (MSE_neg ~1 from random initialization)
- The GC loss provides no new signal that wasn't already implicit in the MSE training dynamics

For GC to improve performance, negatives would need to be "confusable" — close to the data manifold in a structured way that the MSE loss alone cannot discriminate. Random Langevin perturbations don't achieve this for matrix inversion.

---

## Final Recommendation

**Declare IRED-CD contrastive investigation complete. The baseline (MSE-only, no mining) is the best achievable result with this architecture.**

If contrastive shaping is desired for future work:
1. **Reparameterize**: Separate the energy head (for contrastive) from the gradient head (for denoising). Decouple E-magnitude from ∂E/∂x.
2. **Structured negatives for GC**: For matrix inversion specifically, use negatives that are wrong solutions to the same system (e.g., random matrices with similar Frobenius norm to the true inverse but wrong column space).
3. **Score-based approach**: Drop the EBM parameterization entirely. Use a direct score network `s_θ(x, t) = model(inp, x, t)` trained with denoising score matching. No energy, no coupling issues.
