# Direct-energy gradient guard experiment — 2026-08-01

## Motivation

Direct job 36359213 suffered a finite optimizer shock near step 1,517,550. The
global Adam first-moment norm was 0.452 at step 1.50M, rose to 0.726 at 1.55M,
and returned to 0.444 at 1.60M. The disturbance was concentrated in the
timestep-conditioning path and first transformer block, not solely in the
scalar head. No parameter or optimizer tensor became nonfinite.

## Variant proposal

Variant name: `direct-gradient-guard`

Hypothesis: rare heavy-tailed second-order gradients can displace a trained
direct-energy model; clipping only extreme global gradient norms will prevent
such displacement without changing ordinary updates.

Failure mode addressed: finite optimizer shocks during native scalar-energy
training.

EqM compatibility argument: clipping is applied after the unchanged EqM field
loss and its higher-order backward pass. It does not change the transport
target, energy sign, model architecture, or sampling update.

Loss definition: unchanged EqM loss. The treatment applies global L2 gradient
clipping immediately before `AdamW.step()`.

Expected diagnostics if working: finite nonzero head/backbone gradients; the
calibrated threshold lies above ordinary gradients; clipping is rare; matched
short-run loss is not materially worse than the unclipped control.

Expected diagnostics if failing: clipping activates frequently, ordinary loss
is degraded, gradients remain unstable after clipping, or instrumentation
changes throughput enough to confound the comparison.

Minimal test: start both arms from direct checkpoint 1.50M with the same seed,
sampler epoch, global batch, optimizer state, and 2,000-step budget.

Promotion rule: calibration has at least 1,000 finite positive gradient-norm
records and finite nonzero scalar-head/backbone gradients. Lock
`max_grad_norm = 2 * p99(unclipped grad norm)`. The clipped arm must remain
finite, activate clipping on no more than 1% of steps in this ordinary window,
and have final-500-step mean loss no more than 2% above the matched unclipped
control. A zero activation rate is acceptable because this stage tests that the
guard is non-invasive, not that the rare shock deterministically replays.

Kill rule: any nonfinite gradient, more than 1% clipping in the ordinary
window, or greater than 2% loss degradation. Do not silently retune the
threshold.

## Controls

- Negative/null control: the unclipped calibration run from the same 1.50M
  checkpoint.
- Positive plumbing control: focused test applies clipping to a deliberately
  oversized synthetic gradient and verifies that the post-clip norm is bounded.

## Limitations

The original event cannot be replayed exactly because checkpoints do not store
PyTorch/CUDA/DataLoader RNG state. This experiment establishes a measured,
non-invasive guard. Only a longer run can estimate the shock rate.
