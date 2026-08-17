# Corrected-BTM / FD-scalar campaign — report

**Status: DRAFT. Tables A, B, C, D.1 are final. Table D (FID) and Table E await
Phase II-C and a readable ImageNet val split.**

Numbers first, per the mission brief §29. Interpretation is confined to §6-8 and
is labelled with the brief's evidence grades.

---

## The question

> Is the poor late-training behavior of explicit scalar EqM caused substantially
> by the **mixed input-parameter derivative** `grad_theta grad_x phi_theta`
> required by direct gradient matching, rather than by the **nonexistence of a
> useful scalar autonomous transport potential**?

Operationalized: train the corrected conservative BTM population solution
`b* (x) = E[I_dot_t | I_t = x]`, `b_theta = grad_x phi_theta`, `E_theta =
-phi_theta`, using scalar function evaluations and finite differences, and see
whether transport correctness survives while `grad_theta grad_x phi` is avoided
during training.

Five arms: **V** vector BTM; **G** exact scalar gradient matching; **A** exact
Action/Ritz; **D** directional FD (the primary proposal); **F** FD Action/Ritz.
Negative controls: legacy-EqM-target vector and legacy direct scalar.

---

## TABLE A — toy transport (5-atom, 10 seeds/arm, >=100k fresh x0, tc=0.9)

Basin-mass MAE (primary metric, lower better); `R_psi` = weak conservation
residual.

| arm | ring MAE | R_psi | asym MAE | R_psi | stable |
|---|---|---|---|---|---|
| V   vector BTM | 0.0073 | 0.0510 | 0.0104 | 0.0449 | 10/10 |
| G   scalar exact | 0.0076 | 0.0459 | 0.0073 | 0.0439 | 10/10 |
| A   scalar Action/Ritz | 0.0052 | 0.0398 | 0.0043 | 0.0379 | 10/10 |
| D   scalar FD directional | 0.0085 | 0.0559 | 0.0086 | 0.0471 | 10/10 |
| F   scalar FD Action/Ritz | 0.0049 | 0.0412 | 0.0058 | 0.0378 | 10/10 |
| LEGACY vector (neg ctrl) | 0.0197 | 0.7557 | 0.0319 | 0.7231 | 10/10 |
| LEGACY direct scalar (neg ctrl) | 0.0202 | 0.7554 | 0.0321 | 0.7273 | 10/10 |

**Pre-registered gate `MAE_D <= max(0.015, 2 x MAE_V)`: PASS 6/6** (4 scalar arms
x 2 geometries).

Every scalar arm — exact *and* finite-difference — reaches vector-BTM transport
quality. The controls bracket correctly: 2.4-3.7x worse on mass, **15-19x worse
on conservation residual**, confirming the metric is not trivially satisfiable
and that the corrected BTM target is what does the work.

**Structural limitation, stated up front:** the toy *cannot* separate G from D on
stability, because G does not destabilize at this scale. Stability is an
image-scale question, which is why Phase II exists.

---

## TABLE B — FD numerics and estimator variance

**B.1 Accuracy vs step size** (ground truth = exact autograd directional
derivative; frozen late checkpoint; K=4; TF32 off):

| eps_fd | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|---|---|
| rel RMSE | 6.1e-3 | 2.8e-3 | **6.3e-4** | 2.1e-4 | 1.6e-4 | 1.2e-3 |

corr 1.00000, zero non-finite, cancellation ratio <= 4e-3. Textbook U with a
broad plateau 1e-3..1e-2. Operating point **eps_fd = 1e-3** per the
"choose the plateau, not the smallest h" rule.

**FD numerics are NOT the risk.**

**B.2 Estimator variance** (B=8, eps=1e-3, n=64 minibatches):

| arm | noise scale | mean pairwise cos | s/step | peak mem |
|---|---|---|---|---|
| G exact | 0.203 | 0.856 | 0.16 | 6.9 GiB |
| A action exact | 0.341 | 0.768 | 0.23 | 9.9 GiB |
| **D FD K=1** | **6.30** | **0.124** | 0.14 | 6.3 GiB |
| D FD K=4 | 1.95 | 0.333 | 0.30 | 16.6 GiB |

**FD variance is the risk, and it is structural.** The obvious fix does not
work: the accuracy plateau would permit eps=1e-2 (where rel RMSE is *better*),
so it is tempting to blame the ceiling on an h at the noisy edge. It is not. The
dominant variance is `Var_u[u u^T g]` over the Rademacher direction, which is
**independent of h**; float cancellation is already negligible. The variance is
intrinsic to probing a 4096-dim gradient with K rank-1 directions per step. K is
the only lever, and it pays fully.

**First-class negative result for FD:** at 4-GPU DDP scale the cost ordering
**inverts** relative to this single-GPU microbenchmark. FD costs 2.7x (K=1) and
6.6x (K=4) V's wall clock, where the microbenchmark shows D K=1 as the *cheapest*
arm per step. Single-GPU per-step timings do not predict DDP wall clock here.

---

## TABLE C — optimizer behaviour (Phase II-B, ImageNet B/2, batch 256, tc=0.9)

Step-matched at **[0, 22800]**, the horizon the legacy scalar control reached
before EDQUOT killed it. (The arms ran to different horizons, so any "final
window" comparison is confounded with horizon; only the step-matched panel is
readable.)

| arm | g med | g CV | g p95 | g max | clip% |
|---|---|---|---|---|---|
| V vector BTM | 0.183 | 0.744 | 0.665 | 1.13 | 0.00 |
| G scalar exact | 0.240 | 0.755 | **0.613** | 2.88 | 0.00 |
| D1 FD K=1 | 2.605 | 0.360 | 4.574 | 10.22 | 0.00 |
| LEGACY vector (neg ctrl) | 1.767 | 0.868 | 6.667 | 16.59 | 0.00 |
| LEGACY direct scalar (neg ctrl) | 2.971 | 0.772 | 9.755 | 23.51 | 0.00 |

- **G's gradient p95 is LOWER than V's**, while legacy direct scalar's is 15x
  V's. The heavy tail that is the signature of the legacy pathology is absent
  under the corrected target.
- The controls are informative in both directions: legacy *vector* is also bad
  (1.77), so part of the legacy failure is the **target**, not scalarization per
  se. But scalarization costs 1.3x under the corrected target vs 1.7x under the
  legacy one.
- D1 at ~26x V is the FD variance ceiling of B.2, seen in the optimizer.

---

## TABLE D.1 — BTM target matching at eval (Phase II-B)

`target_cosine` = cos(model field, corrected BTM population target b*). Eval
only; never in the training graph.

| arm | cos ALL | cos far | cos near_data | mse/dim |
|---|---|---|---|---|
| V vector BTM | 0.7081 | 0.7180 | **0.6047** | 0.8398 |
| G scalar exact | 0.7046 | **0.7200** | **0.5602** | 0.8634 |
| D1 FD K=1 | 0.6061 | 0.6169 | 0.5276 | 1.0814 |
| D4 FD K=4 | 0.6106 | 0.6134 | 0.5238 | 1.1089 |

**The headline number hides an asymmetry.** G slightly *exceeds* the vector arm
on the FAR field (0.7200 vs 0.7180) and is worse only NEAR the data manifold, by
0.045 — **~13x the far-field gap**, and stable across every step-matched window
from 11.7k to 75k.

This localizes the cost of the conservative parametrization: `b = grad_x phi` is
not uniformly worse at fitting b*, it is worse specifically where the target is
most structured — which is also where sample quality is determined, and the same
region where the legacy EqM target degenerates (`c(gamma) -> 0` as
`gamma -> 1`). That two different scalar formulations lose accuracy in the *same*
region hints the difficulty is a property of representing this field as a
gradient near the manifold, rather than an artifact of either objective.

Stated as a **prediction, not a conclusion**: it says where FID differences
should appear *if* they appear.

D1/D4 plateau at ~0.61 vs 0.70-0.71 — the variance ceiling again, independently.
K=4 is better than K=1 early and still does not close the gap.

`E_mean ~ +1099`, `E_std ~ +1188` confirms the potential is **not calibrated**,
exactly as pre-registered. Only its gradient is claimed to be meaningful.

---

## Sampling operating point (frozen)

FID vs integration horizon, V at epoch 15, 2k samples, stepsize 0.003:

| T | steps | FID |
|---|---|---|
| 0.75 | 250 | 172.94 |
| **3.0** | **1000** | **133.58** |
| 6.0 | 2000 | 137.84 |
| 12.0 | 4000 | 149.26 |

Interior optimum at **T ~ 3.0**. The inherited legacy 250-step default was
costing **39.4 FID**; over-integrating to T=12 gives back 15.7. The self-stopping
interpolant does **not** make the model insensitive to integration horizon —
worth stating because it is easy to assume it does.

---

## TABLE D — FID by arm at matched steps

**PENDING.** Blocked: the FID reference set is the ImageNet val split on
`/n/holylfs06`, which is currently unreadable (see §Infrastructure).

---

## TABLE E — causal interpretation

**PENDING Phase II-C.** Draft grading of what is already supported:

| claim | grade | basis |
|---|---|---|
| A useful scalar autonomous transport potential EXISTS for this problem | **SUPPORTED** | Table A: G, A, D, F all reach vector-BTM transport quality, 6/6 gate PASS, in two geometries. The nonexistence hypothesis is falsified at toy scale. |
| The corrected BTM target — not scalarization — is what fixes the legacy failure | **SUPPORTED** | Table A: legacy vector and legacy scalar are near-identical and both bad (15-19x on R_psi). Table C: legacy vector p95 6.67 vs corrected V 0.665. |
| Scalar training under the corrected target avoids the legacy gradient pathology | **PARTIALLY SUPPORTED** | Table C at matched steps: G's p95 is below V's, legacy scalar's is 15x V's. BUT all measurements are at or before epoch 15 and the legacy pathology onsets ~epoch 55. This is the gap Phase II-C closes. |
| FD training (arm D) is a practical substitute for exact scalar training | **NOT SUPPORTED** | Table B.2 + C + D.1: variance is structural (h-independent), D plateaus at cos ~0.61 vs 0.70, gradient norm ~26x V, and cost is 2.7-6.6x V at DDP scale rather than cheaper. |
| The mixed derivative `grad_theta grad_x phi` is the PRIMARY cause of the legacy failure | **UNRESOLVED** | This is the mission's central question and it needs II-C. Current evidence complicates it: the legacy *vector* control has no scalar parametrization and no mixed derivative, yet is also badly behaved — so the mixed derivative cannot be the whole story. |

---

## Infrastructure (not experimental results)

`/n/holylfs06` developed an intermittent Lustre fault around 2026-08-15:
metadata ops instant, `lfs df` clean on all 33 OSTs, but a subset of reads block
forever in `fstat`, and sequential reads of the dataset tar are also blocked. It
has cost ~38 A100-hours at zero training steps across four Phase II-C attempts.
No healthy replacement exists (the two netscratch copies have the 1000-class
skeleton with empty class dirs; the retired holylabs copy is deleted).

Mitigations now in the harness, both closing silent-failure classes:
- **Data-readiness preflight** — reads 48 files across 24 *random* classes under
  a hard timeout. Random fresh paths are essential: re-reading the same files
  hits cache and makes a sick filesystem look well.
- **Progress watchdog** — invariant: a training job must make measurable
  progress or terminate. Checks the observable (steps advancing), not any
  mechanism, so it catches any stall cause.

Separately, 27 files still defaulted to the deleted ImageNet copy, including the
evaluator the campaign uses. A stale *train* path crashes loudly; a stale *val*
path does not crash at all — it silently rebases every FID. Now resolved through
a fail-closed resolver.

---

## What would falsify the hypothesis

Recorded before II-C reports, so the reading is not chosen after the fact:

- If **G destabilizes past epoch 55** the way legacy direct does (CV -> O(10),
  clip rate rising, flat median with an exploding tail), the mixed-derivative
  explanation is **NOT SUPPORTED**: the corrected target delayed the failure but
  did not remove it.
- If **G stays healthy and V also stays healthy**, the result is SUPPORTED for
  the corrected target but says nothing about the mixed derivative specifically,
  because the legacy *vector* control is already known to be badly behaved
  without any mixed derivative.
- The cleanest discriminator available is therefore **legacy scalar vs legacy
  vector at matched steps past epoch 55** — the pair that differs only in
  scalarization. Both EDQUOT-killed runs need re-running for this.
