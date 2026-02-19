# IRED-CD: Comprehensive Results & Analysis

**Date**: 2026-02-19
**Phase**: Contrastive Divergence Implementation & Debugging
**Key Experiments**: q101–q209

---

## Overall Goal

Replace unstable adversarial negative mining in your IRED-style diffusion EBM with a literature-grounded alternative (CD / IRED-style contrastive shaping) that **improves matrix-inversion performance** by learning a **well-shaped energy landscape** where inference-time gradient descent reliably converges.

---

## Progress We Made

### 1) We Validated the Mechanics of Your Sampler and "Space Consistency"

* **t=999 spike was a false alarm**: your experiments use `diffusion_steps=10`, so `t ∈ [0,9]`, schedule indexing is correct.
* **No space mismatch**: both `E_pos` and `E_neg` are evaluated in **the same x_t / y_t space** at the same timestep `t`. `E_pos` uses `y_t` from `q_sample`; `E_neg` uses Langevin-refined `y_t`.

### 2) We Identified the Core Trap in the Original Energy Parameterization

Your original energy was:
$$E(x,y,t)=|fc4(h)|^2$$

and your denoising prediction depends on $(\nabla_y E)$.

Literature-aligned contrastive objectives (standard CD sign or `softplus(E_pos − E_neg)`) **push $E_{pos}\downarrow$**. In your squared-norm parameterization, that drove `fc4(h_pos) → 0`, which drove **$(\nabla E → 0)$** and produced a **null predictor** (`pred≈0`, MSE stuck at ~1.0). This was empirically confirmed with logs showing `E_pos ↓`, `gradE ↓`, `pred ~ 0`.

Key insight: **In this parameterization, energy magnitude and gradient-field magnitude are tightly coupled**, so "make energy small on positives" can unintentionally kill the denoising signal.

### 3) We Implemented Option B (IRED-Style Logistic Contrastive) Correctly, and Learned Why It Failed

* You added `softplus((E_pos − E_neg)/T)`, reused the same `y_t` for denoising and `E_pos`, gated regularizer/logging correctly, and created a clean ablation config (q206).
* Result: contrastive separated pos/neg (big margins) but **collapsed energy/gradients** under the squared-norm energy, stalling denoising. This matched both theory (coupling) and observed metrics.

### 4) We Implemented the "Cleanest Literature-Aligned Fix" to Avoid the Trap

From the literature (IRED + EBMs + CD/PCD): the clean formulation is a **scalar energy** $E_\theta(x,y,t)$ (no nonnegativity requirement) and compute $(\nabla_y E)$ via autograd.

You implemented a **scalar energy head** and re-ran:

* **q207 scalar baseline**: stable learning, no gradient collapse.
* **q208 scalar + contrastive**: learned but underperformed; contrastive saturated early and left `gradE/pred` smaller than baseline.

This confirmed:

* **Scalar head fixes the collapse pathology**.
* But **contrastive shaping in the current form saturates too quickly** and produces a geometry that slows denoising convergence.

### 5) We Learned About Saturation and Why Simple Knobs Didn't Help

Warmup, lower weight, and timestep windows did not materially change outcomes in early runs:

* `frac_sat ~ 0.997` across contrastive variants
* similar MSE plateaus and reduced `gradE/pred`
* suggests the contrastive objective becomes "too easy" (margin inflation / scale sensitivity / overly strong negatives), then contributes little gradient after early shaping.

Key insight: If the contrastive term saturates immediately, weight/warmup/window won't help much unless you also control **energy scale** or **negative difficulty**.

### 6) We Started Diagnosing Longer-Run Results and Uncovered a Critical Uncertainty

Later polls introduced a new anomaly: **q209 scalar + no-mining** reportedly had **train MSE ~0.009** but **val MSE ~2.5**.

We **did not** accept "scalar head must be abandoned" as definitive because that pattern is much more likely to be caused by:

* train vs val being **different metrics** (denoising noise-MSE vs task/inversion MSE),
* eval pipeline mismatch,
* solver/evaluation inconsistency,
* or metric/logging issues.

Key insight: before drawing architecture conclusions, you need a **metrics contract** and apples-to-apples evaluation between known-good (q101) and new runs.

---

## Key Insights from the Literature That Guided Decisions

* **EBM/CD directionality:** standard CD minimizes `E_pos − E_neg` (data down, model samples up). PCD uses persistent chains. (Hinton CD; Tieleman PCD)
* **IRED's shaping is contrastive, not CD:** it uses contrastive energy separation plus denoising/score supervision to shape an inference-friendly landscape.
* **Scalar energy is the canonical EBM interface:** energy need not be ≥0; only relative differences matter in $p \propto e^{-E}$. Scalar energy + autograd gradients is the clean alignment with EBM training.

---

## What We've Learned Empirically

1. **Squared-norm energy** $(|fc4(h)|^2)$ **+ contrastive/CD sign** can **collapse gradients** and freeze denoising.
2. **Scalar energy head removes the collapse pathology** (pred/gradE remain alive).
3. **Simple contrastive softplus saturates early** and seems to steer you into a worse denoising basin (lower `gradE/pred`, higher MSE plateau) unless scale/difficulty is controlled.
4. Adversarial mining/NCE can show the classic "gets good, then blows up" instability pattern, but you should not call anything "not a bug" without metric/eval confirmations.

---

## Immediate Next Steps (Tight and Decisive)

### 1) Lock Down Metric Definitions and Evaluation Parity

Create and print a clear metrics contract in every run:

* `train_denoise_mse` (noise/score loss, weighted or not)
* `val_task_mse` (final matrix inversion error in y0 space, or whatever you actually care about)
* plus shapes/scales (mean/std) of pred/target

### 2) Cross-Evaluate Checkpoints to Disambiguate q209

* Evaluate a **known-good q101 checkpoint** using the **current q209 validation code**.
* Evaluate q209 using the **historical q101 validation code** (if different).
* Compute "val metric on train data" as a sanity check.

This will tell you whether q209 indicates real generalization failure or an evaluation mismatch.

### 3) Stabilize Infra and Rerun Cleanly

* Address exit 120 (node blacklist / partition selection).
* Address q208 OOM (reduce batch, Langevin steps, replay size, or memory retention).

### 4) If Contrastive Remains a Goal: Change the Objective So It Doesn't Saturate

Only after metrics are verified:

* consider margin/hinge-style contrastive or explicit energy-scale control,
* reduce negative hardness early (fewer Langevin steps, lower replay prob early),
* or apply shaping later (stronger scheduling) once denoising is already improving.

---

## The Style You've Been Using (And What Works Best Going Forward)

Your current style is extremely effective for debugging:

* structured "Executive Summary" blocks
* exact line-level references, shapes, and equations
* compact tables with `Step / MSE / E_pos / E_neg / margin / frac_sat / gradE / pred`
* explicit hypotheses + "smoking gun" checks
* clear experiment IDs, configs, and deltas between runs

Keep doing that. The most helpful additions going forward:

* always label **which metric is being reported** (train vs val, denoise vs task)
* include 3–5 lines of raw logs around any "turning point" (e.g., divergence onset)
* include resolved config printouts at step 0 (flags + effective defaults)

If you paste the exact definition of `train MSE` and `val MSE` for q101 vs q209, we can close the current ambiguity quickly.

---

## Summary of Key Experiments

| Exp ID | Energy Type | Objective | Status | Key Finding |
|--------|------------|-----------|--------|-------------|
| **q101** | Vector (|fc4|²) | Denoising only | ✓ Baseline | val MSE → 0.009 (stable) |
| **q206** | Vector (|fc4|²) | + Contrastive | ❌ Collapsed | Null predictor (pred → 0) |
| **q207** | Scalar | Denoising only | ✓ Stable | No collapse, clean learning |
| **q208** | Scalar | + Contrastive | ⚠️ Plateau | Saturation issue (frac_sat ≈ 1.0) |
| **q209** | Scalar | Denoising only | ❓ Ambiguous | Train MSE ≈ 0.009, val MSE ≈ 2.5 |

**Next Priority**: Disambiguate q209 via cross-checkpoint evaluation (metrics contract)

---

## Document History

- **2026-02-19**: Created comprehensive analysis document capturing full progression from squared-norm to scalar energy, contrastive learning insights, and immediate next steps.
