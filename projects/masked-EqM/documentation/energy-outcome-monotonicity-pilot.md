# Energy-to-outcome monotonicity pilot

## Hypothesis

For a fixed held-out masked-recovery task, lower model scalar energy at the
same initial sampler depth ranks candidates that will have lower terminal
masked-latent reconstruction error under the unchanged GD sampler.  Native
`direct` energy has a stronger rank-to-outcome association than `dot`.

## Why this is falsifiable

The score never receives the clean latent.  The clean latent is used only after
completion to measure masked-region error.  Gradient parameterization ensures a
local spatial descent fact at fixed time; it does not ensure that a low-energy
candidate reaches the correct missing content.

## Frozen pilot protocol

- Checkpoints: retained matched epoch-15 seed-0 `dot` and `direct` EMA files.
- Tasks: the immutable, disjoint ImageNet validation endpoint bank created for
  the geodesic preliminary.  This pilot uses its 128 unique endpoints.
- Pilot subset: first 64 endpoint tasks, fixed seed `20260729`.
- Per task: one shared 50% elementwise keep mask and eight independent Gaussian
  fills of the missing region.  All candidates therefore have the same task and
  sampler depth.
- Score: each model's scalar returned by `get_energy=True`, ranked only within
  that task/model.  No cross-model raw energy comparison is made.
- Outcome: terminal masked-latent MSE after 99 unchanged GD updates at 0.0017.
- Primary statistic: image-clustered mean within-task Kendall concordance of
  score rank with error rank.  Positive is better.  `direct-dot` receives a
  10,000-resample paired bootstrap CI.

## Controls and pilot decision

- Positive control: the candidate's oracle masked-latent error before recovery
  must have mean Kendall > 0.2 against terminal error for `direct`.
- Negative control: independently shuffled energy ranks must have absolute mean
  Kendall < 0.15 for both methods.
- Pilot PASS: direct mean Kendall > 0, both controls pass, and the lower 95% CI
  for paired `direct-dot` is > 0.

This is a single-seed checkpoint-only pilot, not a seed-level paper claim.
Any PASS only authorizes a preregistered matched multi-seed confirmation.
