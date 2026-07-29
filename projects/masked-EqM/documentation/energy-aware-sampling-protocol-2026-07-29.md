# Energy-aware EqM sampling from frozen epoch-15 checkpoints

## Question and claim boundary

This checkpoint-only study tests whether the native scalar available to EqM
can reject overly aggressive descent updates.  It does not retrain a model and
does not claim a general improvement across training seeds.

Primary hypothesis: on the frozen direct epoch-15 checkpoint, per-sample
Armijo backtracking has lower mean FID over a locked, non-saturated grid of
initial step multipliers than fixed-step GD at the same 100 gradient-update
budget.  The paired stratified-bootstrap 95% lower bound of the robust-FID
difference must be positive.  At multiplier 1 it must be non-inferior by a
one-FID upper-bound margin.

## Locked algorithm

The canonical energy is the model's reported scalar `E`; the update field must
satisfy `v = -grad_x E`.  Each active sample proposes `x + alpha*v` and accepts
when `E(trial) <= E(current) - c*alpha*||v||^2`, with `c=1e-4`, `beta=0.5`,
growth `gamma=1.25`, and at most eight backtracks.  Accepted per-sample trial
steps grow only up to their initial-step cap.  Candidate scoring is scalar-only
forward evaluation; it does not construct a higher-order graph.

## Gates and controls

1. **Implementation gate.** On 32 fixed latent/class pairs, direct and dot
   must have cosine(field, -grad(E)) > 0.999 and negligible relative error.
   Failure blocks all sampling evaluation.
2. **Calibration.** A disjoint 1,024-pair bank sweeps multipliers
   `{0.25, 0.5, 1, 2, 4, 8}` for the fixed direct sampler to select a reference
   step and lock four finite, non-catastrophic multipliers.  Armijo uses the
   same starting step.  Calibration samples never enter final intervals.
3. **Evaluation.** A fixed class-balanced final bank starts at 10,000 pairs
   (50,000 only if runtime allows), shared exactly by all arms.  Primary
   direct arms: fixed, Armijo, and a replay schedule whose iteration-wise
   values are calibration-bank median accepted steps.  Dot repeats the same
   comparison; none remains a fixed-step generation anchor.

The fixed sampler is the negative/null control.  Replay is the adversarial
control for a generic schedule: Armijo beating fixed but not replay supports
schedule discovery, whereas Armijo beating replay supports sample-specific
energy feedback.

## Outcomes and accounting

For each arm write FID, KID where available, divergent-trajectory count,
accepted-step distribution, maximum-backtrack rate, final field norm,
gradient evaluations, scalar-only forwards, and wall time.  Bootstrap sample
indices stratified by class and reuse each resample across arms.  Reports must
separate equal-gradient quality from the later compute-matched fixed control.
