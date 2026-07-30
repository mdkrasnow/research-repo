# Direct scalar-energy candidate ranking — three-bank confirmation

## Decision

**FAIL: do not claim that the directly trained scalar is a generally useful
candidate-quality or conditional-compatibility energy.**  Across three
predeclared, independently seeded candidate banks, raw direct energy did track
the independent quality composite, but it did not reliably outperform the
dot-method scalar and did not establish conditional compatibility.  This is a
negative result for the matched epoch-15 checkpoints, rather than evidence
that the scalar head is invalid for sampling.

## Frozen protocol and integrity checks

- Checkpoints were fixed before this confirmation: base `none`, dot, and
  direct epoch-15 seed-0 checkpoints listed in each saved `metrics.json`.
- Each bank used a distinct fixed seed (`20260801`, `20260802`, `20260803`),
  64 held-out source images, 256 disjoint DINO references, and 1,280 candidate
  rows.  Every bank contains 64 rows in each of 20 groups: real, three model
  generations, real and model-specific two-level corruptions, four classifier
  confusable wrong labels per real image, three short-trajectory failures, and
  noise.
- All model scores were computed at `t_eval=1.0`.  The EqM linear path is
  `z_t=t*data+(1-t)*noise`, so this is its clean/terminal endpoint; it was not
  tuned per model.
- The independent target is the mean of rank-normalized DINOv2
  nearest-reference similarity and pretrained ImageNet classifier probability
  of the supplied label.  Neither component uses an EqM score or FID.
- `candidates.csv` preserves candidate metadata, raw direct/dot energies,
  raw base-field norm, and every independent target.  The remote output
  directories additionally preserve all 1,280 candidate PNGs per bank.
- Each within-bank confidence interval resamples complete source-image
  clusters.  The aggregate interval resamples complete fixed banks; with only
  three banks it is deliberately reported as coarse.

## Raw-score aggregate

Positive direct-minus-dot means lower direct energy ranked the candidate pair
better.  Quality correlation is Spearman of `rank(-score)` with the
independent quality target, hence a positive value corresponds to the requested
negative relation between energy and quality.

| endpoint | direct mean [three-bank CI] | dot mean [three-bank CI] | base norm mean [three-bank CI] | direct-dot [three-bank CI] | direct wins |
|---|---:|---:|---:|---:|---:|
| quality rank correlation | 0.530 [0.177, 0.783] | 0.531 [0.185, 0.779] | 0.514 [0.295, 0.634] | -0.001 [-0.008, 0.004] | 2/3 |
| pairwise quality accuracy | 0.686 [0.557, 0.786] | 0.685 [0.560, 0.783] | 0.676 [0.599, 0.718] | +0.000 [-0.003, 0.003] | 2/3 |
| correct-label lower rate | 0.518 [0.500, 0.527] | 0.483 [0.461, 0.504] | 0.497 [0.453, 0.527] | +0.035 [-0.004, 0.066] | 2/3 |
| corruption increases | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.833 [0.793, 0.871] | +0.000 [0.000, 0.000] | 0/3 |

The three individual raw direct quality correlations were `0.631`, `0.177`,
and `0.783`; dot's were `0.629`, `0.185`, and `0.779`.  Thus the apparent
quality signal is shared to essentially numerical equality with dot, and its
large bank-to-bank variation prevents a strong generalization claim.

Direct's only consistent contrast is against base field norm on synthetic
corruption monotonicity.  That contrast is insufficient: dot is exactly as
monotone, and neither scalar separates direct from dot on the primary quality
or pairwise tests.  The conditional endpoint is weak: direct is at chance in
one bank (0.500) and its two positive direct-minus-dot bank differences do not
give an aggregate lower confidence bound above zero.

## Conclusion against the success criterion

The direct scalar does **not** pass the requested criterion.  It satisfies a
limited property—raw energy rises on this fixed latent corruption path and has
a positive association between `-E` and the independent quality composite—but
it fails the required discriminating tests: no reliable direct-over-dot
improvement, no clear conditional-label advantage, and no stable superiority
over the field-norm baseline on quality.  Do not deploy it as an independent
sample-quality or label-compatibility score for candidates outside its own
trajectory.

## Reproduction and artifacts

Run `energy_candidate_ranking/evaluate.py` with one of:

- `configs/energy_candidate_ranking_confirmation.json`
- `configs/energy_candidate_ranking_confirmation_seed2.json`
- `configs/energy_candidate_ranking_confirmation_seed3.json`

Then aggregate the three `metrics.json` files with
`energy_candidate_ranking/aggregate_confirmation.py`.  The checked-in output
bundle is `results/direct_energy_campaign/candidate_ranking_confirmation/`:
`three_bank_metrics.json`, this table in `three_bank_summary.md`, three raw
candidate CSVs, and distributions, quality scatterplots, and
corruption-level plots for each raw score.  The rendered candidate images are
retained at `/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/energy_candidate_ranking/`
under the three `confirmation_*` directories.
