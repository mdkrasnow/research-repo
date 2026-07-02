# Dynamics probe summary

good=205 garbage=154; 5-fold OOF, seed=0

| feature set | dim | OOF AUROC | within-norm AUROC |
|---|---|---|---|
| norm_only | 1 | 0.574 | 0.578 |
| endpoint_only | 3 | 0.579 | 0.571 |
| scalar_only | 13 | 0.672 | 0.668 |
| FULL | 61 | 0.701 | 0.700 |

- FULL beats endpoint_only by +0.129 (within-norm)
- FULL beats norm_only by +0.121
- FULL beats scalar_only by +0.032

top features:
  - dot_ds0: +0.919
  - norm_ds1: -0.917
  - dot_ds5: -0.858
  - norm_ds2: -0.821
  - late_norm_slope: -0.554
  - l2_ds3: +0.530
  - norm_oscillation: +0.521
  - early_norm_slope: +0.516

## VERDICT: BORDERLINE
Dynamics within-norm 0.700 in 0.65-0.70 — ambiguous.
