# Native scalar-energy EqM campaign — final report

## Verdict

**NEGATIVE at pilot scale.** The native scalar-energy model is technically
valid and trainable, but this campaign found no evidence that it improves the
tested EqM behavior sufficiently to justify full multi-seed training.

## Implementation verification

The model emits a scalar energy by conditionally normalizing final transformer
tokens, projecting each token to one scalar, and summing contributions. Its
field is `-grad_x E`. Focused CPU tests and GPU smoke jobs verified the sign,
shape, higher-order training gradient, direct CFG behavior, checkpoint guard,
and detached sampler lifecycle.

## Learnability and cost

Fixed-batch direct overfit passed: loss fell 19.21 to 15.73 and cosine rose
0.192 to 0.254. In the corrected 256-latent matched pilot, direct loss fell
22.37 to 14.88 and cosine rose 0.218 to 0.593. Direct used 2.63 GB and 5.79
steps/s, essentially matching dot (2.62 GB, 5.80 steps/s) and costing about
2.8x vanilla (1.28 GB, 16.27 steps/s).

## Sampling

Direct sampling was finite in all pilot probes. The measured practical
step-size interval was eta=0.0015 through 0.006; eta=0.012 stayed finite but
overshot in latent norm. NFE trajectories were finite through 250 updates,
but terminal latent norm increased at the long horizon, so no lower-NFE or
more-reliable-descent advantage was established.

## Robustness

At off-trajectory radius 2.0, direct relative field change was 0.04542 versus
0.04298 for none and 0.04377 for dot. It is therefore stable but does not
support the central conservativeness-improves-off-path-stability hypothesis.

## Decision

The full three-seed result-bearing run and expensive unseen-corruption study
were not launched. The available checkpoints are deliberately short pilot
models; their recovery/FID results would not be scientifically meaningful, and
the pre-registered evidence criterion for promotion was not met.

See `stage10_decision.md`, `stage6_stepsize_report.md`, `stage7_nfe_report.md`,
and `stage8_off_trajectory/report.md` for supporting artifacts.
