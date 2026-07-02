# Dynamics probe summary

good=750 garbage=750; 5-fold OOF, seed=0

| feature set | dim | OOF AUROC | within-norm AUROC |
|---|---|---|---|
| norm_only | 1 | 0.542 | 0.538 |
| endpoint_only | 3 | 0.681 | 0.672 |
| scalar_only | 13 | 0.718 | 0.714 |
| FULL | 61 | 0.704 | 0.743 |

- FULL beats endpoint_only by +0.071 (within-norm)
- FULL beats norm_only by +0.205
- FULL beats scalar_only by +0.029

top features:
  - norm_ds2: -2.047
  - dot_ds0: +1.964
  - l2_ds4: +1.806
  - dot_change: +1.729
  - norm_ds1: -1.597
  - norm_ds4: +1.424
  - dot_ds5: -1.234
  - late_norm_slope: -1.213

## VERDICT: PROMISING
Dynamics within-norm 0.743 in 0.70-0.80 — worth a small metacognitive sampler pilot (see METACOGNITIVE_RESCUE_SPEC.md).
