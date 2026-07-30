# Direct scalar-energy candidate-ranking pilot — 2026-07-30

## Decision

**Superseded by repaired v2 below.**  The original eight-per-group pilot
completed successfully (SLURM `36266037`, exit `0:0`), but had correlated-row
bootstrap and candidate-coverage defects.  It is retained only as a plumbing
record.

## Repaired v2 decision

**FAIL — do not promote to a larger candidate-ranking run.**  Job `36267576`
(exit `0:0`) used 16 source-image clusters, each model's own generated
corruption ladder, short-trajectory model-specific failures, and source-cluster
bootstrap confidence intervals.  Direct raw energy: quality Spearman `-0.151`
(`[-0.333, 0.026]`), pair accuracy `0.447` (`[0.380, 0.502]`), and correct-label
lower rate `0.312` (`[0.125, 0.562]`).  Dot also had perfect corruption
monotonicity, so direct's `1.0` offers no discriminating advantage.

## Fixed protocol

- One immutable 88-candidate bank, seed `20260730`, evaluated at `t_eval=1.0`.
  This is the repository's clean/terminal convention:
  `z_t=(1-t)noise+t*data`.
- Candidates: held-out reals; matched-label samples from base, dot, and direct;
  two fixed latent corruption levels of real and generated candidates;
  wrong-label real pairs; heavily corrupted failures; and pure latent noise.
- Quality was independent of all EqM scores: the mean of rank-normalized DINOv2
  nearest-disjoint-reference cosine similarity and pretrained ImageNet
  classifier probability of the supplied label.  Individual components and all
  raw scores are in `candidates.csv`.
- Raw direct and dot scalar energies and raw base vector-field L2 norm were
  evaluated on exactly the same candidate tensors and labels.  Lower is the
  hypothesized better value for all three.

## Results

| score | quality Spearman (95% CI) | pair accuracy (95% CI) | correct label lower (95% CI) | real corruption increases (95% CI) |
|---|---:|---:|---:|---:|
| direct energy | -0.022 [-0.222, 0.200] | 0.490 [0.419, 0.566] | 0.375 [0.122, 0.750] | 1.000 [1.000, 1.000] |
| dot energy | -0.028 [-0.236, 0.193] | 0.489 [0.419, 0.562] | 0.375 [0.125, 0.750] | 1.000 [1.000, 1.000] |
| base field norm | +0.084 [-0.119, 0.307] | 0.533 [0.452, 0.605] | 0.750 [0.375, 1.000] | 1.000 [1.000, 1.000] |

The two scalar energies share perfect monotonicity on this simple
real-to-noise latent interpolation, but neither has nonzero-quality ranking
evidence.  The direct scalar also has no conditional advantage and does not
outperform the baselines.  This answers the stated question negatively for the
tested held-out candidate distribution; it is not evidence that a different
checkpoint, calibration, or target would succeed.

## Artifacts

Remote immutable result directory:
`/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/energy_candidate_ranking/pilot/`

It contains `candidates.csv` (metadata, raw scores, and independent targets),
`metrics.json`, candidate PNGs, and score-distribution and quality-rank plots.
No FID was computed or reused as a per-sample target.
