# WFB-EqM Stage 3 postmortem: no-Adam Forward-Backward Gauss-Newton training

**Status**: FBGN (CG) arm complete + analyzed. WFB and direct arms completing (see
`documentation/wfb-eqm-stage3-proposal-2026-08-13.md` for the pre-registered design,
promotion/kill rules).

## Summary

Per the pre-registered kill rule: *"if all three arms show near-zero accept rate or
flat/worsening probe loss -> the backtracking-line-search globalization itself is
insufficient at this checkpoint's curvature scale; would need... a genuine Levenberg-
Marquardt adaptive-damping scheme (increase `rho`/lambda on repeated rejection, not just
shrink `eta`)."* This is exactly what happened for FBGN, with the mechanism now
precisely characterized.

## FBGN (alpha=1.0, CG solver) — full 300-step run, job 38909657

**Mechanism**: completely clean. `accept_rate=1.0`, **zero skips of any kind** across
all 300 steps (no `not_descent_direction`, no `backtrack_exhausted`) -- the CG fix (see
below) fully resolved the earlier fixed-k=96 Lanczos pathology (3/20 theoretically-
impossible skips in the pre-fix smoke, job 38900551).

**Trajectory**: net WORSENING. `probe_loss` 10.625 -> 14.881 (+4.257) over 300 steps.
Same-batch `L_before` showed growing INSTABILITY late in the run despite the clean
Armijo mechanism -- spiked to 40.7 (step 285) and **130.6** (step 292), each "accepted"
via a correspondingly huge `actual_delta_L` (Armijo only requires sufficient decrease
relative to the CURRENT point; it has no view of where that point came from or where
it's heading).

**Root cause, precisely characterized from the run's own `metrics.jsonl`**:
- `r_norm` (residual/field-error magnitude) stayed roughly BOUNDED the entire run
  (530-800 throughout) -- the model's raw predictions never diverged in the way a
  simple "loss exploding" story would suggest.
- `lambda_max` (top eigenvalue of the mixed input-parameter Gram operator `A = M M^T`,
  i.e. the LOCAL CURVATURE the Gauss-Newton step solves against) trended up
  **~36x on average** (first-10-step avg 1.7e7 -> last-10-step avg 6.1e8) and **spiked
  as high as 1.23e10** (a ~19,400x range over the run, min 6.3e5 to max 1.23e10).
- Every same-batch loss spike coincides EXACTLY with a `lambda_max` spike, not with
  `r_norm` growth. This is the mechanism: FBGN's step is curvature-blind by
  construction. Armijo checks "did loss decrease relative to *this step's* local
  quadratic model" -- it has no mechanism to notice or penalize that the model is
  drifting into regions of ever-sharper local curvature. `rho` (the damping-to-
  `lambda_max` ratio inside the Lanczos/CG solve) was held FIXED at 1e-4 for the whole
  run -- it damps proportionally to *current* curvature at each step, but does nothing
  to prevent `lambda_max` itself from growing across steps.

**Backbone-vs-head localization** (postmortem diagnostic, job 38942662, comparing the
run's own saved step100/step300 checkpoints via `block_subspace_iteration_theta` on a
fixed 4-batch probe pool): the curvature growth is ENTIRELY a backbone phenomenon.
Median top-1 singular value: backbone 6597 -> 10101 (**1.53x growth**), head 434 -> 414
(**0.95x, flat/slight shrink**). This is the SAME structural signature as the original
Stage A postmortem on the Adam-trained instability (`documentation/postmortem-stageA-
topk-subspace-2026-08-11.md`, CLAUDE.md summary: *"energy head is not the dominant
contributor... backbone spike/control amplification scales monotonically with tail
severity"*) -- now confirmed under a COMPLETELY DIFFERENT optimizer (no-Adam Gauss-
Newton via CG + Armijo line search, not Adam). Strong evidence this is a property of
the backbone architecture under double-backward/energy-gradient training itself, not an
artifact of any particular optimizer.

(Caveat: the postmortem's probe pool differs from the exact training batches, so its
1.53x doesn't numerically match the live power-iteration's ~36x average / ~19,000x peak
-- the backbone-vs-head SPLIT is the robust finding, not the exact magnitude.)

## The CG fix (context, already committed separately)

Stage 3's FIRST smoke (fixed k=96 Lanczos, job 38900551) hit `not_descent_direction`
skips on 3/20 steps -- a condition that is THEORETICALLY IMPOSSIBLE in exact arithmetic
(`A(A+lambda I)^{-1}` is PSD, so `r . q_alpha >= 0` always). Traced to k=96 Lanczos
truncation (same mechanism Stage 2.6a already characterized via the `rho_m = -ell_m`
residual certificate). Fixed by implementing matrix-free CG (Hestenes-Stiefel) for the
shifted system `(A+lambda I) u = r` -- adaptive-tolerance (stops at a target residual
ratio, e.g. 1e-3) rather than fixed-k truncation, since `A+lambda I` is SPD and CG is
the textbook exact method for that setting. CPU-validated against direct solve + full-k
Lanczos; on the real model, converged in 15 iterations to a properly-bounded gradient
(field_gain=0.983) vs the fixed-k=12 measurement of 2.83-3.61. This fix is durable and
worked for the entire 300-step run (zero skips of any kind) -- the divergence documented
above is a SEPARATE, later-stage finding about the algorithm's lack of curvature
control, not a recurrence of the truncation bug.

## What this rules in / rules out

- **Ruled out**: "FBGN doesn't train because of numerical truncation error." The CG fix
  makes the mechanism airtight; the algorithm still diverges. This is a genuine
  algorithmic finding, not a bug.
- **Ruled out (partially)**: "the checkpoint is simply too far into the instability
  regime for any first-order-model step-size rule." The backbone-specific localization
  argues against a diffuse "everything is broken" story and toward a specific,
  addressable mechanism (backbone curvature drift).
- **Ruled in**: FBGN (and by extension any pure-Gauss-Newton-via-line-search scheme
  applied to this architecture) needs an explicit curvature-control mechanism beyond
  Armijo's local sufficient-decrease check -- i.e. genuine Levenberg-Marquardt adaptive
  damping (`rho` increases when curvature grows or steps underperform their predicted
  reduction, not just a fixed ratio + eta-backtracking), exactly as the pre-registered
  kill rule anticipated.

## Next step (not yet started)

Implement adaptive damping: after each step, compare the realized `lambda_max` (or the
realized-vs-predicted reduction ratio, already computed as part of Stage 2.6b's `rho`
diagnostic) against its value at the START of the run (or a moving reference), and
increase `rho` (hence `lambda` in the shifted solve) when curvature is growing faster
than some threshold -- classic trust-region radius shrink, but keyed to curvature
growth specifically rather than just loss-prediction mismatch. Given the backbone
localization, an alternative/complementary angle: restrict the FBGN correction to
backbone parameters only, leaving the head on a separate (simpler) update rule, since
the head shows no curvature-growth pathology at all.

## WFB (alpha=0.5) and direct (alpha=0.0) — pending

Both still running at time of writing (jobs 38903397, 38904393); ~283/300 and ~275/300
steps respectively. Will update this section once complete.
