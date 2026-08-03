# Direct checkpoint forensics: 1.45M--1.60M

## Result

The degradation is a discrete optimizer-state shock, not evidence that the
direct scalar architecture was gradually becoming worse before step 1.50M.
The same event is visible independently in fixed-batch field metrics and a
matched FID-500 probe.

| Step | EMA loss, mean over t | EMA cosine | EMA norm ratio | EMA HVP norm | FID-500 |
|---:|---:|---:|---:|---:|---:|
| 1.45M | 11.3066 | 0.7334 | 0.7152 | 6.2989 | 175.27 |
| 1.50M | 11.3114 | 0.7334 | 0.7154 | 6.3480 | 174.74 |
| 1.55M | 13.2552 | 0.6814 | 0.6303 | 4.8668 | 245.71 |
| 1.60M | 12.3223 | 0.7047 | 0.6965 | 5.7122 | 228.76 |

The 1.45M and 1.50M checkpoints are essentially tied. Between 1.50M and
1.55M, FID-500 worsens by 70.96, mean field loss rises by 1.94, and mean
field-target cosine falls by 0.052. Partial recovery occurs by 1.60M, but both
field metrics and FID remain substantially worse than the pre-shock state.

FID-500 is a noisy localization probe and is not comparable in absolute value
to the established 2,000- or 50,000-sample protocols. Its matched discontinuity
is nevertheless large enough to corroborate the field diagnostics.

## Localization by corruption level

The damage is strongest at intermediate/high corruption-path values. From
1.50M to 1.55M, EMA loss changes by +2.80 at t=0.6, +3.55 at t=0.7, and +3.83
at t=0.8. EMA cosine changes by -0.071, -0.092, and -0.103 respectively. This
is not a uniform scalar rescaling.

## Optimizer-state evidence

The shock interval produces broad early-backbone and optimizer-moment movement:

- x-embedder raw displacement is 3.54 times its preceding-interval value.
- block 0 raw displacement is 1.70 times its preceding-interval value.
- block 0 Adam first-moment displacement is 3.08 times its preceding value.
- block 0 Adam second-moment displacement is 13.53 times its preceding value.
- scalar-head Adam second-moment displacement is 21.02 times its preceding value.

This agrees with the separately measured 65.13 global-gradient outlier and
supports a broad second-order optimizer shock rather than a scalar-head-only
failure.

## Negative findings

- Directional Hessian norms decrease at 1.55M rather than exploding.
- Every recorded fixed-t GD update lowers direct energy; energy-increase
  fraction is exactly zero for raw and EMA weights at all four checkpoints.
- Samples and fields remain finite.
- Every sampling iterate is detached (`grad_fn` absent), so no trajectory graph
  accumulates.

These results do not support curvature regularization, a sampler line search,
or changing the sum-based scalar readout as the first ablation.

## Ablation decision

The justified first ablation is the preregistered global gradient guard at
`max_grad_norm=6.871409955`. It directly intercepts the observed rare event,
activated on only 1/2,000 matched steps, and did not degrade short-run loss.

Do not add head normalization, mean pooling, curvature penalties, a different
sampler, or a head-specific learning rate based on these data. If a long guarded
continuation still suffers a shock, the next discriminating ablation would be
clipping plus a narrowly specified Adam-moment recovery rule. Moment reset is
not yet warranted because preventing the outlier upstream should prevent its
moment contamination.

## Artifacts

- `checkpoint_forensics.png`: field, curvature, and sampling trajectories.
- `checkpoint_state_deltas.png`: layerwise parameter and Adam displacement.
- `fid500_localization.png`: matched quality discontinuity.
- `direct_*.json`: raw checkpoint-level diagnostics.
- `state_deltas.json`: layerwise state comparisons.
- `fid500/direct_*.json`: matched FID localization results and sample grids.
