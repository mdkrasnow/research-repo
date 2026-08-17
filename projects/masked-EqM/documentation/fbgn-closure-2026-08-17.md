# FBGN closure (mission brief §24)

**Status: CLOSED — negative result. No further FBGN compute is authorized.**

The brief asked for *closure, not rescue*. This document states what FBGN was,
what was established, what was retracted, why it is being stopped, and what is
worth keeping. It is written so the thread does not have to be re-derived by
anyone (including a future me) who finds the partial results and mistakes them
for an unfinished promising direction.

---

## 1. What FBGN was trying to do

EqM `--ebm direct` trains a scalar energy `E_theta(z)` whose field is
`s_theta(z) = -grad_z E_theta(z)`, fit to the EqM target under squared error.
The object that makes this hard is the mixed input-parameter Jacobian
`M = d(grad_z E)/d theta`; ordinary training uses `g = M^T r` with
`r = s_theta(z) - y`.

Stage 1 established that the mixed-Jacobian amplification is **real**: the raw
gradient spike went from 2.44x the control to 0.95x under whitening. FBGN was
the attempt to exploit that — replace `g = M^T r` with the damped Gauss-Newton
direction `p = -M^T (M M^T + lambda I)^{-1} r`, solved by residual-controlled CG.

## 2. What was actually established

**The optimizer works, on the objective it is given.** At 300 steps, batch 8,
Armijo backtracking: accept rate 1.000, zero skips, and FBGN removes 66-74% of
its own minibatch's residual, monotonically.

**And it destroys the objective we care about.** Deterministic held-out probe
over the same 300 steps:

| arm | probe start -> end | delta | median R | median eta |
|---|---|---|---|---|
| direct (alpha=0) | 10.6244 -> 10.8324 | +0.208 | 0.991 | 3.8e-6 |
| WFB (alpha=0.5) | 10.6244 -> 11.2145 | +0.590 | 0.911 | 2.5e-3 |
| FBGN (alpha=1, CG) | 10.6245 -> 14.8811 | **+4.257** | 0.528 | 1.19 |

Note the negative control: **raw direct also worsens the probe.** The failure
runs through the entire alpha family, ordered exactly as step aggressiveness is
ordered. FBGN is the extreme point of a family-wide effect, not a special defect
of Gauss-Newton. Without that control the result would have been misread as "GN
is broken."

## 3. The mechanism (Stage 3A, job 38985448)

Two hypotheses were pre-registered:

- **H1** — Gauss-Newton local-model failure (FBGN drops `sum_i r_i grad^2 r_i`).
  Repair would be genuine LM damping / shorter steps.
- **H2** — stochastic minibatch over-solving: the local model is *right* and the
  minibatch is *wrong*. Damping cannot help.

The discriminator was designed to be **eta-independent**, because that is the
only way to separate them: the infinitesimal transfer `d_V = grad L_V . p`
measured on an independent trust bank. If `d_V >= 0`, then by first-order Taylor
**no step size and no damping can rescue the direction**, and H2 is decided
outright regardless of trust-region effects.

| quantity | value |
|---|---|
| FBGN cosine with independent-batch gradient `C_V` | **0.00108** |
| raw-direct minibatch gradient `C_V` | 0.0395 |
| FBGN `d_V` descent rate (eta-INDEPENDENT) | **4/8** |
| trust bank worsened at `eta*/8` | 8/8, +0.0629 |
| damping x10 / x100 -> `C_V` | -0.0031 / -0.0083 |

**H1 is present but is a trust-region artifact** — `D_B` falls as O(eta) and
`R_B -> 1`, exactly what a *correct* linearization does. Nothing is wrong with
the GN model; `eta* ~ 1.3` was simply far outside its trust region.

**H2 is what survives eta -> 0**, and it is causal. The pre-registered H1 repair
failed its own falsifier: damping fixed the local model and *simultaneously
killed* independent-batch descent.

Mechanism, stated plainly: at a converged checkpoint the B=8 GN system carries
almost no population signal, and `(A + lambda I)^{-1}` whitening amplifies
precisely the small-curvature directions that only those 8 images constrain — a
further 36x alignment loss. **FBGN removes 66-74% of its own batch's residual
while retaining 0.1% of the population gradient direction.** Certification makes
this worse, not better.

That is the entire reconciliation of "100% Armijo acceptance + monotone
same-batch reduction" with "monotone probe damage." Both are true. They are
statements about different objectives.

## 4. The kill (Stage 3B)

Stage 3A's mechanism makes a sharp, falsifiable prediction: if B=8 is *merely
signal-starved*, then `C(B) = cos(g_B, g_ref)` must rise **faster than the
sqrt(B) of pure noise-averaging**. Pre-registered gate: exponent > 0.65 means
batch size is a real lever; ~0.5 or below fires the KILL.

Measured, n=8 reps:

| B | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| C(B) | +0.005 ± 0.040 | +0.166 ± 0.053 | +0.231 ± 0.077 | **-0.033 ± 0.102** | +0.089 ± 0.070 |

The script reported "exponent 0.784 > 0.5 -> batch size is a real lever." **That
report must not be used to pass the gate**, and this is the single most
important sentence in this document. R^2 = 0.286, the series is non-monotonic,
and B=32 -> B=64 is a **5.9-sigma DECREASE** with a negative mean at B=64. A
monotone power law is rejected by that comparison alone, so the fitted exponent
summarizes a model the data reject. Fitting an exponent to a series that is not
a power law yields a number, not a measurement.

Honest read: **alignment stays <= 0.23 and draw-noise-dominated at every
affordable B.** The KILL fires. This does not authorize Stage 3C or any further
FBGN compute.

(Arm 2, the stacked-GN arm, crashed on an empty `micro_batches` list. It is not
worth fixing: it was contingent on arm 1 showing batch size to be a lever, and
arm 1 shows the opposite.)

## 5. Why this closes rather than pauses

Under the repo's stop conditions, FBGN trips at least three:

- **Diagnostic signal saturated or near-zero** — `C_V = 0.00108`.
- **The improvement requires post-hoc reinterpretation** — the only surviving
  "success" metric (own-batch residual reduction) is the one the mechanism
  explains as *causing* the damage.
- **The same failure repeated across two reasonable HP settings** — damping x10
  and x100 both made `C_V` worse, not better.

And it fails at the level of theory, not tuning: `d_V >= 0` on an independent
bank means the direction is not a descent direction for the population objective
**at any step size**. There is no hyperparameter under this parametrization that
recovers it, because the deficiency is first-order.

## 6. What survives and is worth keeping

The scientific content that outlives the method:

1. **The mixed input-parameter Jacobian amplification is real and is
   localized to the backbone** (backbone sigma_1 1.53x vs head 0.95x). Stage 1's
   whitening result stands. This is precisely the premise the corrected-BTM /
   FD-scalar campaign now tests from the other direction — by *avoiding*
   `grad_theta grad_x phi` during training rather than preconditioning it.
2. **Minibatch-certified optimization is dangerous near convergence.** Any
   method that solves its own minibatch subproblem to high accuracy should be
   assumed to be over-solving unless it is checked against an independent bank.
3. **Method-level lesson:** an eta-independent discriminator was what made this
   decidable. The eta-dependent quantities (`rho_lm`, `R_B`, Armijo acceptance)
   are all confounded with step length and were consistent with *both*
   hypotheses.

## 7. Corrections issued during this thread (do not re-derive)

- **"lambda_max growth caused the divergence" is RETRACTED.** Probe damage is
  ~78% complete by step 25 while lambda_max grew for 275 more steps — the
  timing is decoupled. And with `lambda = rho * lambda_max`, the operator
  `A(A + lambda I)^{-1}` is invariant under `M -> cM` *by construction*, so
  uniform curvature growth cannot change the step at all.
- **The logged reduction ratio cannot decide the mechanism.** `rho_lm` is
  confounded with step length: FBGN takes `eta ~ 1.19`, direct takes
  `eta ~ 3.8e-6`. Any smooth model is accurate as `eta -> 0`, so a low R for the
  arm taking far longer steps is what *both* hypotheses predict.
- **Never read the FBGN mechanism off `fbgn100`/`fbgn300`.** `||g_V||` is
  74.6/192.2 there vs 1.204 at `start`, and raw-direct `C_V` is 0.98/0.88 — the
  model is broken enough that almost anything descends. Their apparent H1
  signature measures distance-from-optimum, not mechanism. Only the checkpoint a
  failure *began* from can explain it.
- **CG's own reported residual understated the true one ~1.9x** (0.0086 vs
  0.0159). Recompute `||r - (A + lambda I)u|| / ||r||` from a fresh operator
  application before calling any direction certified.

## 8. Pointers

- Audit: `documentation/wfb-eqm-stage3-audit-2026-08-13.md`
- Stage 3A report: `documentation/wfb-eqm-stage3a-report-2026-08-13.md`
- Superseded interpretation (kept for provenance):
  `documentation/wfb-eqm-stage3-postmortem-2026-08-13.md`
- Memory topic: `memory/topics/wfb-eqm-fbgn-stage3.md`
- Successor thread: `memory/topics/btm-fd-scalar-campaign.md`
