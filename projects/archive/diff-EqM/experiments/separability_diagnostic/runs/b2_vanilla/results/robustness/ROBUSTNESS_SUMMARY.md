# Robustness summary — label-threshold sweep

label quantiles swept: [0.2, 0.25, 0.3, 0.35]
scores present: ['s1', 's2', 's3', 's4', 's5']

| label_q | best independent (within-norm) | best norm-coupled | s4 sanity | usable bins |
|---|---|---|---|---|
| 0.2 | 0.616 | 0.545 | 0.635 | 5 |
| 0.25 | 0.609 | 0.539 | 0.627 | 5 |
| 0.3 | 0.602 | 0.539 | 0.614 | 5 |
| 0.35 | 0.602 | 0.539 | 0.603 | 5 |

- independent ≥0.80 at 0/4 thresholds (0%)
- mean s4 label-sanity = 0.620 (<0.55 => LABEL-BROKEN)
- max independent across thresholds = 0.616

## VERDICT: WEAK
Independent 0.60–0.80 — real but sub-actionable; see dynamics_probe.py for combined signal.
