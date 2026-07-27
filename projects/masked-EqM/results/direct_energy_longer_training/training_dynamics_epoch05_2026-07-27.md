# Seed-0 training dynamics through epoch 5

This report parses the completed continuation logs for the matched seed-0
`none`, `dot`, and `direct` runs. Each log contains 3,203 measurements through
step 200,150. Slopes are least-squares fits of logged loss against optimizer
step; they are not finite differences of individual minibatches.

## Summary

| arm | first 10% loss | final 10% loss | change | final 20% slope / step | final 500 slope / step | final 1,000 slope / step | steps/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 11.0473 | 10.7209 | -0.3264 | -1.22e-6 | -1.71e-7 | -8.59e-7 | 8.89 |
| dot | 11.4197 | 11.1182 | -0.3015 | -1.12e-6 | -1.62e-7 | -8.08e-7 | 4.75 |
| direct | 11.5086 | 11.1456 | -0.3629 | -1.16e-6 | -7.34e-7 | -1.08e-6 | 4.74 |

All three arms show sustained downward movement over the full run. Direct has
the largest first-to-last-tenth loss reduction and retains a clearly negative
1,000-point slope. The 500-point windows are close to zero because minibatch
noise dominates at the end; the final 100 points are not a reliable trend
estimate.

## Loss at epoch-scale step boundaries

| arm | ~40k | ~80k | ~120k | ~160k | ~200k |
|---|---:|---:|---:|---:|---:|
| none | 11.0278 | 10.8186 | 10.7551 | 10.6972 | 10.7739 |
| dot | 11.3965 | 11.1864 | 11.1403 | 11.1137 | 11.1561 |
| direct | 11.3846 | 11.3915 | 11.3259 | 11.1683 | 11.0378 |

The logged minibatch at step 200k is noisy, so these individual points should
not be read as epoch averages. The smoothed windows and FID checkpoints are
the stronger evidence: direct closes the training gap by late training and
reaches FID 134.74 at epoch 3 and 128.02 at epoch 5, versus dot 136.28 and
127.72, respectively.

## Throughput

`none` runs at about 8.89 steps/sec, while the explicit-energy arms run at
about 4.75 steps/sec. The roughly 1.9x overhead is consistent with the input
gradient construction; the direct run does not show a throughput anomaly.

## Artifacts and provenance

- Parsed logs: `projects/masked-EqM/slurm/logs/longer_{none,dot,direct}_seed0_interleaved2_*.log`
- Machine-readable summary: `training_dynamics_epoch05_2026-07-27/summary.json`
- Plot: `training_dynamics_epoch05_2026-07-27/training_dynamics.png`
- Checkpoint FIDs: 2,000 samples and 250 sampling steps per arm/checkpoint.

This remains a one-seed diagnostic. Seed-1 and seed-2 continuations are still
required before making a replicated scientific claim.
