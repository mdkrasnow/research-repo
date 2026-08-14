---
name: btm-fd-scalar-campaign
description: Corrected-BTM scalar potentials trained by finite differences — the toy gate passed 6/6 in two geometries, FD numerics are validated at real ImageNet B/2 scale, and FD estimator VARIANCE is the identified risk
status: active
---

# Corrected-BTM / FD-scalar campaign (started 2026-08-13)

Thread: `projects/masked-EqM/`, branch `btm-fd-scalar`. Supersedes the FBGN
optimizer-rescue track (see [[wfb-eqm-fbgn-stage3]], now closed negative).

## The hypothesis

The late-training pathology of explicit scalar EqM (`--ebm direct`) is caused
substantially by the **mixed input-parameter derivative** `d_theta d_x phi` that
direct gradient matching requires — not by the nonexistence of a useful scalar
autonomous transport potential. Train the *corrected* conservative BTM solution
using only scalar energy evaluations + central finite differences and see
whether transport correctness survives while the optimization pathology goes.

Arms V (vector) / G (scalar exact) / A (action exact) / D (FD directional) /
F (FD action), plus legacy-EqM negative controls. G and D are constructed to
approach the **same** conservative population solution as `h -> 0`, which is
what makes a divergence in their *training behaviour* isolate an optimization
mechanism.

## Paper facts (arXiv:2608.01692v2, recovered from the full HTML)

- `div(nu b) = mu0 - mu1`, `nu = int_0^1 mu_t dt` (Theorem 1).
- §2.7: EqM's eq.(16) pairs the LINEAR interpolant with target `c_t (x1-x0)`,
  which is `E[Idot | I_t]` for **no** interpolant; it coincides with the
  consistent loss only at `c_t == 1`.
- Appendix H eq.(57) self-stopping interpolant; `alpha(tc)=(1-tc)/(1+tc)` and
  `alpha_dot(tc)=-2/(1+tc)` on both branches (C^1), `Idot_1 = 0`.
- **`tc` is NOT numerically specified anywhere in the paper.** Swept; we froze
  `tc = 0.9` on the toy (worst-arm MAE monotone in tc: .0111/.0100/.0093/.0080).
- Five-atom benchmark: `p=(.30,.30,.15,.15,.10)`, EqM schedule `c_t=(1-t)^0.8`,
  reference BTM MAE 0.005 vs EqM 0.102. **Atom coordinates are not published** —
  ours are documented benchmark VARIANTS.
- BTM correction is worth FID 1.87 vs 1.90 at XL/2 — small but free.

## Toy result: GATE PASS 6/6, Outcome A (2026-08-13)

10 seeds/arm, 100k fresh x0 per model, exact `grad phi` as evaluation drift.
Median mass MAE, ring geometry (and asymmetric geometry in parens):

| V | G | A | D | F | legacy vec | legacy scalar |
|---|---|---|---|---|---|---|
| .0073 (.0104) | .0076 (.0073) | .0052 (.0043) | **.0084 (.0086)** | .0049 (.0058) | .0197 (.0319) | .0202 (.0321) |

**Every scalar arm — exact AND finite-difference — reaches vector-BTM transport
quality.** The corrected target is what does the work. Both legacy controls are
2.4-3.7x worse on mass and ~15-17x worse on the weak conservation residual
(0.71-0.75 vs 0.033-0.059).

- **The conservation residual is the geometry-independent discriminator.** Mass
  MAE depends on benchmark geometry; the divergence-equation violation does not.
- **Symmetry hides the legacy bias.** A symmetric ring cancels most of the
  schedule-induced reweighting (legacy 0.020); unequal radii/gaps restore it
  (0.032, toward the paper's 0.102) while every BTM arm stays flat. If you build
  an atomic benchmark to expose a mass-REALLOCATION bias, make it asymmetric.
- **The toy structurally CANNOT separate G from D on stability**, because G does
  not destabilize at this scale. That separation is an image-scale question.

## Image-scale de-risking (frozen late `direct` checkpoint 2825000.pt)

**FD numerics are NOT the risk.** Exact autograd directional derivative as
ground truth, K=4, TF32 off; three independent jobs agree:

| eps_fd | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|---|---|
| rel RMSE | 6.1e-3 | 2.8e-3 | 6.3e-4 | 2.1e-4 | 1.6e-4 | 1.2e-3 |

corr 1.00000, zero non-finite, cancel ratio <= 4e-3. Textbook U with a broad
plateau 1e-3..1e-2 — 4-5 significant digits at the energy scale training
actually reaches. Operating point `eps_fd = 1e-3`.

**FD VARIANCE is the risk.** B=8, eps=1e-3, n=64 minibatches:

| arm | noise scale | mean pairwise cos | s/step | peak mem |
|---|---|---|---|---|
| G exact | 0.203 | 0.856 | 0.16 | 6.9 GiB |
| A action exact | 0.341 | 0.768 | 0.23 | 9.9 GiB |
| D FD K=1 | **6.30** | **0.124** | **0.14** | **6.3 GiB** |
| D FD K=4 | 1.95 | 0.333 | 0.30 | 16.6 GiB |
| F FD K=1 | 12.65 | 0.049 | 0.21 | 9.7 GiB |

D at K=1 is ~31x noisier than exact G but is the CHEAPEST arm measured (faster
and lower memory than exact). K=4 recovers the predicted 1/K. At the real global
batch of 256 the further 32x averaging puts D(K=1) near exact-G-at-B=8; whether
that suffices is what Phase II-A measures. Both K=1 and K=4 are in the pilot.

Weak conservation residual on the legacy-trained checkpoint: ~1.09 linear /
1.14 quadratic (~100%) — the legacy model does not satisfy the divergence
equation, measured on our own model. Expected-failure negative control.

## Load-bearing gotchas (each would have produced a wrong answer)

1. **`LabelEmbedder` resamples classifier-free-guidance dropout on EVERY forward
   in train mode, and it is the ONLY stochasticity in this architecture** (timm
   Attention/Mlp built with `drop=0`). Uncontrolled, `phi(z+hu)` and `phi(z-hu)`
   get INDEPENDENT conditioning and the FD numerator measures dropout noise
   instead of the derivative — **an error that grows without bound as h -> 0**.
   Fix: sample the mask once per step and freeze the embedder
   (`image_losses.frozen_label_dropout`). `fb_direct/exact_hvp.py`'s
   `compute_field_direct` already did the equivalent `_predrop_labels`; the
   knowledge existed in the repo but was written down nowhere findable.
2. **Never DOWNcast in an FD subtraction.** An unconditional `.float()` demoted
   fp64 references to fp32, inflating FD error ~20x at eps=1e-3 and destroying
   the O(h^2) signature. Promote low precision only.
3. **Fused SDPA kernels have no double-backward.** Any arm differentiating
   through `grad_z E` must disable flash/mem-efficient SDPA as train.py does.
4. **Never store n full B/2 gradients** (130M params, ~1 GiB each in fp64 — it
   OOMs). Exact streaming identities: `E|g-gbar|^2 = E|g|^2 - |gbar|^2`, and
   mean pairwise cosine `= (|sum_i g_i/|g_i||^2 - n)/(n(n-1))`.
5. macOS `multiprocessing.Pool` deadlocks when a `Lock` crosses a spawned Pool
   initializer — use independent sharded processes with per-shard output.
6. The repo's existing `sample_gd.py` (`x <- x + eta*b(x)`, `t` zeroed by
   `uncond=True`) is ALREADY the autonomous BTM solver — no new sampler needed
   for evaluating BTM arms.

## Status

Toy stage COMPLETE (gate pass, both geometries). Phase II-A launched
2026-08-13: 12 ImageNet B/2 jobs, 4 arms x 3 seeds, 20k steps, 4 GPUs each.
The decisive measurement is whether G destabilizes late while D stays stable
AND learns a comparable exact evaluation-time field.
