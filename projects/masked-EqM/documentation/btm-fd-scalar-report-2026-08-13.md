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

## 1. Toy stage (Experiment I) — five-atom benchmark

*Pending completion of job 39062340; tables and gate verdict inserted on completion.*

Frozen `t_c = 0.9` from the stage-1 sweep (worst-arm criterion over
`{0.5, 0.7, 0.8, 0.9}`; worst-arm mean MAE 0.0111 → 0.0100 → 0.0093 → 0.0080,
monotone in `t_c`). FD step size frozen at `eps_fd = 3e-3` from the stage-0 calibration
ladder run at four points along a real training trajectory (eps* stayed 3e-3 from step
250 to step 6000 — the plateau does not move materially as the toy energy scale grows).

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
can rescue it. The pre-registered damping repair failed its own falsifier. Stage 3B
(job 39064005) is running purely to attach a measured `C(B)` exponent to the write-up.
No further FBGN compute is authorized beyond it.
