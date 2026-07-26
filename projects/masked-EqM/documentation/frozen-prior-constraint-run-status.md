# Frozen-prior constraint run status

## Smoke recovery

* Job `35326768` (seas_gpu) reached the evaluator, wrote all 16 Gaussian and
  Bernoulli none/hard records, then failed at mixed loading because the
  prelaunch manifest named a pruned checkpoint.  It ran for 00:05:47 and
  exited `1:0`; this is a manifest-provenance failure, not a model result.
* The corrected manifest uses the verified 1:1 seed-0 checkpoint from job
  `31549055`; unavailable 1:1 mixed seed-1/2 paths and the available 1:2
  checkpoint were excluded.  The retried bundle is resume-safe and will reuse
  the already completed Gaussian/Bernoulli records.
* The bundle's static walltime is now 12 hours, matching `seas_gpu`; the prior
  2d23h directive required a submission-time override.

## Pre-launch validation

* Immutable evaluator revision before the latest recovery: `9d09905`.
* Locked manifests: pilot `eb0e5f3343be7f74c052a8b5fd62cc062b905c64deb85b7fdaa5448e71daf90f`; final `34a4540e0a0739a68590d2d9e0f1ca73e11e64fa01275d12d7fdbfe3d72a90ec`.
* Focused tests: 8 passed locally, including exact tensor-level disabled-projection GD/NAG regression and batched-mask metric shape coverage.

The smoke will be resubmitted from the corrected immutable revision.

## Completed smoke

* `35332640` — `seas_gpu`, `COMPLETED` 2026-07-26 14:27 EDT, 00:04:53,
  exit `0:0`, immutable revision `c5ab95c`. The six expected files contain
  96/96 finite records (16 paired examples × 3 arms × none/hard). Hard
  projection had observed latent MSE = 0 and maximum invariant deviation = 0
  for every arm. The unconstrained controls had nonzero observed errors, as
  expected. No metric or loading error occurred.
