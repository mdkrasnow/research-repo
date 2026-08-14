# Corrected-BTM scalar potentials via finite differences — campaign report

**Date**: 2026-08-13 · **Branch**: `btm-fd-scalar` · **Status**: IN PROGRESS

> **Hypothesis**: the poor late-training behaviour of explicit scalar EqM is caused
> substantially by the mixed input–parameter derivative `d_theta d_x phi` that direct
> gradient matching requires, rather than by the nonexistence of a useful scalar
> autonomous transport potential.

---

## 0. What has been established so far (numbers first)

### 0.1 FD estimator accuracy at real ImageNet B/2 scale — RESOLVED, POSITIVE

Frozen late-training `--ebm direct` checkpoint (`2825000.pt`, the epoch-80-scale model
the original instability was diagnosed on), exact autograd directional derivative as
ground truth, `K=4`, TF32 disabled. **Two independent jobs (39062911, 39063725):**

| eps_fd | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|---|---|---|---|
| h (mean) | 4.5e-4 | 1.4e-3 | 4.5e-3 | 1.4e-2 | 4.5e-2 | 0.135 | 0.452 | 1.35 |
| rel RMSE (job 1) | 6.3e-2 | 2.2e-2 | 6.1e-3 | 2.8e-3 | 6.3e-4 | 2.1e-4 | 1.6e-4 | 1.2e-3 |
| rel RMSE (job 2) | 4.6e-2 | 2.8e-2 | 6.5e-3 | 2.7e-3 | 7.2e-4 | 2.6e-4 | 1.6e-4 | 1.3e-3 |
| correlation | 0.999 | 0.9996 | 0.99998 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| cancel ratio | 1.2e-6 | 3.7e-6 | 1.2e-5 | 3.6e-5 | 1.2e-4 | 3.6e-4 | 1.2e-3 | 3.6e-3 |
| nonfinite | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Textbook U-shape: catastrophic cancellation on the left, `O(h^2)` truncation on the
right, and a **broad plateau from 1e-3 to 1e-2** where the estimator agrees with the
exact directional derivative to 4–5 significant digits. Not a knife-edge.

**Conclusion**: FD *numerics* are not what will decide this campaign. Chosen operating
point `eps_fd = 1e-3` (inside the plateau, an order of magnitude of cancellation
headroom below it, so it stays valid if the energy scale drifts during training).

### 0.2 Gradient-estimator variance at image scale — the real risk to Arm D

Same frozen checkpoint, B=8, `eps_fd=1e-3`, n=64 independent minibatches per arm,
statistics computed by exact streaming identities (no gradients retained):

| arm | E‖g‖² | ‖E g‖² | noise scale | mean pairwise cos | s/step | peak mem |
|---|---|---|---|---|---|---|
| G exact gradient matching | — | — | **0.203** | **0.856** | 0.16 | 6.9 GiB |
| A exact Action/Ritz | — | — | 0.341 | 0.768 | 0.23 | 9.9 GiB |
| D FD directional, K=1 | — | — | **6.30** | **0.124** | **0.14** | **6.3 GiB** |
| D FD directional, K=4 | — | — | 1.95 | 0.333 | 0.30 | 16.6 GiB |
| F FD Action/Ritz, K=1 | — | — | **12.65** | **0.049** | 0.21 | 9.7 GiB |

Reading:

- Arm D's per-minibatch gradient at `K=1` is **~31× noisier** than the exact arm's, and
  `K=4` buys almost exactly the `1/K` variance reduction that averaging `K` independent
  probe directions predicts (6.30 → 1.95, ratio 3.2).
- The mechanism is dimensional, not numerical: a single directional probe measures one
  scalar projection of a `d = 4096`-dimensional residual.
- This is **not automatically fatal**. These numbers are at `B=8`; the real recipe uses
  global batch 256, a further 32× averaging, which puts Arm D at `K=1` at roughly the
  noise scale exact-G already has at `B=8`. Whether that is enough is exactly what
  Phase II-A measures.
- Arm D at `K=1` is the **cheapest arm measured** — faster per step and lower peak
  memory than exact gradient matching. A stability win at `K=1` would be free.
- Arm F is the noisiest by a wide margin (`cos = 0.049`), consistent with the a-priori
  concern that the endpoint-difference estimator carries much more variance.

### 0.3 Weak conservation residual on the legacy checkpoint — expected failure

`R_psi = E_nu[grad psi . grad phi] - (E_mu1 psi - E_mu0 psi)`, 32 frozen probes:
median normalized residual **1.094** (linear) and **1.14** (quadratic) — i.e. ~100%.
The existing `--ebm direct` model, trained with the legacy `c_t`-scaled EqM target, does
**not** satisfy the BTM divergence equation. This is the paper's §2.7 claim measured
directly on our own model, and it is a *negative control for the diagnostic*, not a
finding about FD: a legacy-trained model is supposed to fail this test.

---

## 1. Toy stage (Experiment I) — five-atom benchmark: **GATE PASS 6/6**

Frozen `t_c = 0.9` from the stage-1 sweep (worst-arm criterion over
`{0.5, 0.7, 0.8, 0.9}`; worst-arm mean MAE 0.0111 → 0.0100 → 0.0093 → 0.0080,
monotone in `t_c`). FD step size frozen at `eps_fd = 3e-3` from the stage-0 calibration
ladder run at four points along a real training trajectory (eps* stayed 3e-3 from step
250 to step 6000 — the plateau does not move materially as the toy energy scale grows).

### Table A — toy transport (ring geometry, 10 seeds/arm, 100k fresh x0 per model)

| arm | mass MAE (mean±std) | median | unresolved | R_weak (median) | stable |
|---|---|---|---|---|---|
| V vector BTM | 0.0086 ± 0.0039 | 0.0073 | 0.0011 | 0.049 | 10/10 |
| G scalar exact | 0.0081 ± 0.0030 | 0.0076 | 0.0010 | 0.044 | 10/10 |
| A action exact | 0.0051 ± 0.0021 | 0.0052 | 0.0009 | 0.042 | 10/10 |
| **D FD directional (K=1)** | 0.0087 ± 0.0034 | **0.0084** | 0.0010 | 0.059 | 10/10 |
| F FD action (K=1) | 0.0055 ± 0.0024 | 0.0049 | 0.0010 | 0.043 | 10/10 |
| — EqM legacy (vector) | 0.0202 ± 0.0053 | 0.0197 | 0.0016 | **0.752** | 10/10 |
| — EqM legacy (scalar) | 0.0208 ± 0.0053 | 0.0202 | 0.0022 | **0.745** | 10/10 |

For scale, the paper reports BTM 0.005 and EqM 0.102 on its own (unpublished) atom
coordinates. Our BTM arms land at 0.005–0.009, matching its BTM figure. Our legacy
control is milder than its 0.102 — the ring geometry is symmetric, which cancels much of
the schedule-induced reweighting (confirmed directly in Table A' below) — but it is
still 2.4–2.7× worse than every BTM arm,
and on the **weak conservation residual it is ~15× worse** (0.75 vs 0.04–0.06).

The residual is the sharper discriminator, and it is the more meaningful one: it
measures, in our own harness, the paper's §2.7 claim that the `c_t`-scaled target does
not enforce `div(nu b) = mu0 - mu1`. Mass MAE depends on benchmark geometry; the
divergence-equation violation does not.

### Table A' — the same comparison on the ASYMMETRIC geometry (10 seeds/arm)

The ring is symmetric, and symmetry cancels much of the schedule-induced reweighting.
Repeating the comparison with unequal radii (2.2–3.8) and unequal angular gaps tests
that directly:

| arm | ring median | **asym median** | asym R_weak | stable |
|---|---|---|---|---|
| V vector BTM | 0.0073 | 0.0104 | 0.044 | 10/10 |
| G scalar exact | 0.0076 | 0.0073 | 0.042 | 10/10 |
| A action exact | 0.0052 | 0.0043 | 0.033 | 10/10 |
| **D FD directional** | 0.0084 | **0.0086** | 0.042 | 10/10 |
| F FD action | 0.0049 | 0.0058 | 0.040 | 10/10 |
| — EqM legacy (vector) | 0.0197 | **0.0319** | 0.716 | 10/10 |
| — EqM legacy (scalar) | 0.0202 | **0.0321** | 0.712 | 10/10 |

Breaking the symmetry moves the legacy control 1.6× worse (0.020 → 0.032, toward the
paper's 0.102) while **every BTM arm stays flat**. That is the predicted signature: the
legacy target's error is a mass-*reallocation* bias that a symmetric geometry hides,
whereas the BTM arms are transporting correctly in both. Gate 4 strengthens from 2.34×
to **3.71×**, and the conservation residual separates by ~17× (0.71 vs 0.033–0.044).

### Pre-registered gate for Arm D — all six conditions

| # | condition | result |
|---|---|---|
| 1 | FD estimator numerically validated | PASS — calibration ladder, plateau chosen not minimum h |
| 2 | stable across seeds | PASS — 10/10 |
| 3 | `MAE_D <= max(0.015, 2*MAE_V)` | PASS — 0.0084 ≤ 0.015 |
| 4 | decisively beats legacy-EqM control | PASS — 2.34× on the ring, **3.71× on the asymmetric geometry** (needs > 2×) |
| 5 | no mixed input–parameter derivative | PASS — guard, with Arm G as positive control |
| 6 | exact field transports to the right modes | PASS — unresolved 0.0010 |

### Interpretation: Outcome A of the pre-registered decision table

V, G **and** D all succeed. The corrected BTM target is what does the work: *every*
scalar arm, exact or finite-difference, reaches vector-BTM transport quality, while both
legacy controls fail. Finite differences are **viable** at this scale but are **not
required** for it — the toy cannot separate G from D on stability because **G does not
destabilize here**. The late-training pathology is an image-scale phenomenon, and
separating G from D on it is the entire purpose of Experiment II.

---

## 2. Implementation defects found and fixed (each one would have produced a wrong answer)

1. **fp64 demotion in the FD subtraction.** `fp32_subtract=True` unconditionally cast
   evaluations to fp32, inflating FD error ~20× at `eps=1e-3` and destroying the
   `O(h^2)` convergence signature. Now promotes low precision only.
2. **Label-dropout resampling.** `LabelEmbedder` resamples classifier-free-guidance
   dropout on *every* forward in train mode, and it is the only stochasticity in this
   architecture. Uncontrolled, `phi(z+hu)` and `phi(z-hu)` are evaluated under
   independent conditioning and the FD numerator measures dropout noise instead of the
   directional derivative — an error that *grows without bound as `h -> 0`*.
   `frozen_label_dropout()` samples the mask once per step, for all arms, so the paired
   G-vs-D comparison shares conditioning too.
3. **Fused SDPA has no double-backward.** Any arm differentiating through `grad_z E`
   must disable flash/mem-efficient SDPA exactly as `train.py` does.
4. **Storing n full B/2 gradients OOMs.** Replaced with exact streaming identities
   (`E‖g-ḡ‖² = E‖g‖² - ‖ḡ‖²`; mean pairwise cosine `= (‖Σ ĝ_i‖² - n)/(n(n-1))`),
   verified against brute force to 1e-13.
5. **Stage 3B's `@torch.no_grad` on `bank_loss`** made `E.sum()` a leaf while
   `compute_field_direct` needs the input-gradient graph — the cause of both prior
   Stage 3B failures.

---

## 3. FBGN closure

Stage 3A already decided the mechanism: at the healthy start checkpoint the certified
CG-Gauss-Newton direction has `C_V = 0.00108` with the independent-batch gradient and is
an ascent direction on 4/8 batches — `eta`-independent, so no step size and no damping
can rescue it. The pre-registered damping repair failed its own falsifier. Stage 3B (job 39064005) has now run, and **the thread closes as a negative result.**

Measured `C(B) = cos(g_B, g_ref)` at the start checkpoint, n=8 reps per B:

| B | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| mean | +0.005 | +0.166 | +0.231 | **−0.033** | +0.089 |
| SD | 0.040 | 0.053 | 0.077 | 0.102 | 0.070 |
| z vs 0 | +0.4 | +8.8 | +8.5 | −0.9 | +3.6 |

The script printed `C(B) ~ B^0.784 ... exponent materially ABOVE 1/2: batch size is a
real lever`. **That conclusion is not supported and is not adopted.** `R^2 = 0.286`, and
the series is non-monotonic: `B=32 -> B=64` is a **5.9-sigma decrease** with a negative
mean at B=64. A monotone increasing `C(B)` is rejected by that single comparison, so the
fitted exponent summarizes a model the data reject rather than measuring a scaling law.
The pre-registered threshold (`exponent > 0.65`) nominally passes, but its precondition
— that `C(B)` follows a power law — is violated, so it cannot be read as a pass.

This is the same failure mode already recorded for this thread ("a large reduction ratio
`R_B` at a large step is not evidence of anything"): a canned interpretation string
applied to a statistic whose preconditions were never checked.

Honest read: alignment with the population gradient stays small (≤ 0.23) and
draw-noise-dominated at every affordable B. Arm 2 (stacked GN at `B_model=32`) crashed
on an empty micro-batch list and is deliberately **not** being fixed — it was contingent
on arm 1 establishing the lever. No further FBGN compute is authorized.
