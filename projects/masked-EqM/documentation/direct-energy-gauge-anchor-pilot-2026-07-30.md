# Variant proposal: gauge-anchored scalar-energy ranking

Variant name: per-label zero-latent anchored direct energy.

Hypothesis: direct gradient matching leaves an additive, class-dependent
`b(y)` unconstrained.  Raw `E(x,y)` can therefore fail cross-label ranking even
if its within-label energy differences are meaningful.  `E(x,y)-E(0,y)` removes
that fixed offset without changing the learned field.

Failure mode addressed: the repaired raw-score pilot failed quality and
conditional ranking while both scalarizations were perfectly monotone along the
trivial corruption path.

EqM compatibility argument: subtracting a fixed-in-`x` scalar changes neither
`-grad_x E` nor sampling.  The raw value remains saved and primary; anchored
values are an explicitly labelled secondary identifiability diagnostic.

Loss definition: none; this is checkpoint-only evaluation.

Expected diagnostics if working: direct anchored score has negative
quality Spearman CI, pair-accuracy CI above 0.5, correct-label-lower CI above
0.5, and exceeds dot anchored and base norm on those endpoints.

Expected diagnostics if failing: anchored and raw rankings remain near chance,
or dot benefits identically; do not claim a useful direct scalar.

Minimal test: one new 16-source fixed candidate bank, exact v2 checkpoints,
terminal `t=1`, source-cluster bootstrap, no tuning.

Promotion rule: only if every direct anchored endpoint above clears its stated
CI and direct exceeds both baselines; otherwise stop this scalar-ranking line.

Kill rule: any failed endpoint, or equal improvement for dot, kills the
"useful calibrated direct scalar" claim for these checkpoints.
