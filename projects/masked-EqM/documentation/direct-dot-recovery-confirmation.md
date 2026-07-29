# Direct versus dot masked-recovery confirmation

## Motivation and status

The epoch-15 seed-0 energy-to-outcome pilot did **not** support the energy-rank
claim.  It did, however, show an exploratory paired terminal masked-latent MSE
advantage for direct: dot-minus-direct `+0.3403`, bootstrap 95% CI
`[+0.2712,+0.4264]` across 64 held-out image tasks.  This document preregisters
a separate outcome claim; it does not reinterpret the failed energy result.

## Hypothesis

Matched native-scalar `direct` checkpoints have lower terminal masked-recovery
error than matched `dot` checkpoints under the unchanged sampler.

## Frozen protocol

- Three independently trained, matched direct/dot seed pairs.  Seed 0 is the
  existing epoch-15 pair.  Seeds 1 and 2 must be trained with identical data,
  schedule, and checkpoint epoch before their paths are added to the manifest.
- Immutable held-out ImageNet validation task bank; no training image or pilot
  task selection after seeing recovery outcomes.
- Per image: one seeded 50% elementwise keep-mask and eight seeded Gaussian
  fills.  Both variants receive the same task/candidate tensors.
- Unchanged GD sampler: 100 steps, step size 0.0017.  No hard constraint.
- Primary endpoint: per-image mean terminal masked-latent MSE, paired as
  `dot - direct` (positive favors direct).
- Secondary: decoded masked-region MSE and full-image LPIPS, reported without
  replacing the primary endpoint.

## Controls and decision

- VAE encode/decode round-trip floor is recorded per bank, not used as an arm.
- Candidate/tensor identity hashes must match between arms; an independently
  shuffled arm pairing is a negative analysis control.
- Within each seed, image-cluster bootstrap produces a paired CI.  Across the
  three seed effects, use a two-sided one-sample t test and report its CI.
- **PASS:** mean seed effect > 0, two-sided `p < .05`, and every seed effect is
  positive.  **FAIL:** otherwise.  No retune or selective epoch choice follows
  a failure.

## Promotion and kill rules

PASS permits a paper-facing recovery claim and a single robustness extension.
FAIL retires the direct-energy recovery-improvement story; no additional
surrogate-energy diagnostic is proposed as a rescue.
