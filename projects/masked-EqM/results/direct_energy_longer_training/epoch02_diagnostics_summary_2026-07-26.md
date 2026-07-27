# Epoch-2 diagnostic summary

Updated: 2026-07-26 20:42 ET

This report evaluates the fixed seed-0 epoch-2 checkpoints using the same
encoded batch, corruption noise, and nine timestep values (`t=0.1...0.9`).
The evaluator uses `train=True` so the reported backbone gradients include the
second-order path required by field matching. Results are from diagnostic job
`35387742`; output was written to home03 because the holylabs quota was full.

## Fixed-batch metrics

| arm | mean loss | mean field cosine | mean field/target norm ratio | mean head grad | mean backbone grad |
|---|---:|---:|---:|---:|---:|
| none | 12.793 | 0.672 | 0.677 | 1.23 | 0.70 |
| dot | 12.979 | 0.671 | 0.729 | 2.23 | 1.00 |
| direct | **12.956** | **0.671** | 0.723 | 2.48 | 1.41 |

Direct is slightly below dot on this fixed-batch loss at epoch 2 while having
indistinguishable mean directional alignment. Its backbone gradient remains
finite and stronger than both controls. The result does not show a scalar
head gradient collapse or a directional conservativeness defect.

## Timestep behavior

The three arms track closely through `t=0.1...0.8`. At `t=0.9`, direct and dot
have norm ratios around 0.84, while none is around 0.58; this endpoint has a
different target scale under the repository's `c(t)` convention and should be
reported separately rather than averaged as if it were an ordinary interior
timestep.

## Generation probes

Completed 2,000-sample, 250-step FIDs:

| arm | checkpoint | FID |
|---|---|---:|
| none | seed 0, epoch 2 | 152.8576 |
| dot | seed 0, epoch 2 | 144.6914 |
| direct | seed 0, epoch 2 | pending (`35387519`) |

The none and dot values are not a final matched multi-seed result. Direct's
epoch-2 FID is still running. Epoch-3/5 FIDs will be launched after the
continuation jobs produce those checkpoints.

## Current training state

- none seed 0 reached epoch 5 (`35337285`).
- dot seed 0 is still training; epoch 3 checkpoint exists (`35337286`).
- direct seed 0 is still training; epoch 3 checkpoint exists (`35337288`).
- home03 free space is approximately 8.7 GB.

## Interpretation

At epoch 2, direct has caught up to dot on fixed-batch field loss and matches
both controls on field direction. This is evidence for delayed optimization,
not evidence of a structural scalar-energy bottleneck. The generation claim
remains open until direct FID and the epoch-3/5 checkpoints are evaluated.
