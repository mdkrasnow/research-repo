# Full-screen training diagnostics (2026-07-26)

The user-directed matched-FID continuation was stopped as an analysis priority. The two already-submitted probe jobs (`35290253`, `35290255`) completed, but the remaining probes are not being launched or used for a three-seed FID claim.

## Source and method

This report parses the nine completed one-epoch EqM-B/2 training logs under `/n/home03/mkrasnow/direct_energy_full_retry`. Logs report loss and steps/sec every 50 optimizer steps (approximately 40,000 updates). For each run, the first and final tenth of recorded points are averaged to reduce minibatch noise.

## Results

| arm | seeds | first-tenth loss | final-tenth loss | change | final steps/sec |
|---|---:|---:|---:|---:|---:|
| none | 0,1,2 | 13.122 | 11.103 | -2.019 | 4.35, 4.35, 4.34 |
| dot | 0,1,2 | 13.352 | 11.492 | -1.860 | 2.21, 2.30, 2.30 |
| direct | 0,1,2 | 13.847 | 11.603 | -2.244 | 2.31, 2.30, 4.72 |

The direct run is approximately 0.11 loss above dot and 0.50 above none in the final tenth. All arms show sustained downward movement; none shows the lowest final field loss, dot is intermediate, and direct is not numerically divergent.

The direct seed-2 throughput (4.72 steps/sec) is an outlier relative to direct seeds 0/1 (2.30–2.31) and should not be interpreted as a stable performance estimate without checking node/kernel conditions.

## What this does and does not establish

* **Established:** all three arms completed the same one-epoch training budget without NaNs or scheduler/OOM failure; direct is learnable at full scale; direct has a measurable second-order-autograd cost in two of three runs.
* **Not established:** field-target cosine, field/target norm ratio, energy statistics, per-timestep behavior, or gradient norms for these full-screen runs. The full-screen logger did not record those quantities.
* The earlier 200-step fixed-batch direct test remains the relevant gradient-flow evidence: loss 19.21→15.73, cosine 0.192→0.254, finite head/backbone gradients, and stable memory.

## Interpretation

The loss ordering is consistent with direct scalar energy being viable but somewhat harder to optimize or differently scaled than the vector-output controls. It is not evidence that the scalar-energy hypothesis is false: the runs are short, the loss is not a generation metric, and direct was not compared with matched FID as requested.

## Recommended next diagnostic

Use the existing fixed-batch evaluator on the nine checkpoints (same encoded batch and corruption seed) to record field cosine, norm ratio, energy mean/std/range, head and backbone gradient norms, and memory. This directly tests whether the full-screen loss gap is caused by field scale/alignment rather than architecture failure. Do not launch longer or multi-seed training until that audit is complete.
