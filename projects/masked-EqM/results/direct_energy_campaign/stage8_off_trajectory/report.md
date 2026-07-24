# Stage 8 — Off-trajectory field stability

Each arm used the matched 1,000-step pilot checkpoint at `t=0.5`, the same
eight encoded examples and four random L2 perturbation radii. These are
stability diagnostics, not off-path target errors: EqM does not define a
unique supervised field target at perturbed points.

At radius 2.0:

| Arm | Field cosine to on-path field | Relative field change |
|---|---:|---:|
| none | 0.999070 | 0.04298 |
| dot | 0.999042 | 0.04377 |
| direct | 0.998972 | 0.04542 |

All probes were finite. Direct is stable, but the central hypothesis that an
exact scalar potential improves local off-trajectory consistency is not
supported at this pilot scale: it is marginally less invariant than both
controls on this metric. No claim about recovery-to-data is made here.
