# Energy monotonicity of EqM parameterizations

## Scientific question and hypothesis

Does native scalar-energy parameterization improve held-out noise-to-data energy monotonicity over dot scalarization while remaining non-inferior to ordinary vector-field EqM by a 0.01 absolute margin?

Epoch 8 is the sole primary checkpoint. Direct passes only when the paired image-cluster bootstrap lower CI for `direct-dot` is >0 and the lower CI for `direct-none` is >-0.01.

## Corruption path and confirmed sign

`z_gamma = gamma*x + (1-gamma)*epsilon` with 21 evenly spaced points. The repository trains its returned field toward `(x-epsilon)c(gamma)`, the sampling direction. Dot differentiates `z·f(z)`; direct returns `-grad E`. Therefore canonical decreasing energy uses fixed signs `none=-1`, `dot=-1`, `direct=+1` on the requested raw line integrals. These signs were derived from source before evaluation.

## Evaluation bank

2048 held-out images, 2 Gaussian draws per image, seed 12345, no augmentation, deterministic ordering, training-matched center crop/normalization/VAE encoding. Ground-truth labels are supplied; CFG is disabled.

## Checkpoints and field definitions

6 EMA checkpoints were validated for variant, epoch, architecture, dataset configuration, and corruption setup. The full manifest with paths, hashes, steps, parameter counts, and run metadata is in `checkpoint_manifest.json`.

- none: existing vector output, canonical path gradient = its negative.
- dot: input gradient of per-sample `sum(z*f(z))`, never the raw vector.
- direct: input gradient of exactly one native scalar per sample.

## Primary metric and bootstrap

Trajectory-level strict pairwise ordering accuracy over all 210 gamma pairs. Ties are failures. The 10,000-replicate paired bootstrap samples 2,048 image clusters and always includes both associated noises, using seed 23456.

## Epoch-8 primary results

| variant | ordering | 95% CI | adjacent | perfect | Spearman | drop | ties | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 1.000000 | [1.000000, 1.000000] | 1.000000 | 1.000000 | -1.000000 | 13493.8 | 0 | 0 |
| dot | 1.000000 | [1.000000, 1.000000] | 1.000000 | 1.000000 | -1.000000 | 13327.5 | 0 | 0 |
| direct | 1.000000 | [1.000000, 1.000000] | 1.000000 | 1.000000 | -1.000000 | 13220 | 0 | 0 |

| comparison | difference | paired 95% CI |
|---|---:|---:|
| direct - dot | 0.000000 | [0.000000, 0.000000] |
| direct - none | 0.000000 | [0.000000, 0.000000] |
| dot - none | 0.000000 | [0.000000, 0.000000] |

## Verdict: FAIL

one or both preregistered confidence-interval conditions fail.

## Checkpoint learning curves

Available epochs in the requested 1-8 window are [1, 8]; epochs with no retained checkpoint are [2, 3, 4, 5, 6, 7]. All retained checkpoints use the same frozen bank. Machine-readable values are in `per_epoch_summary.csv` and paired differences in `paired_differences.csv`; all pre-8 epochs are diagnostic and were not used for selection.

## Numerical validation

Dot and direct raw scalar differences were compared with trapezoidal raw line integrals on 256 trajectories at 21 and 101 points.

```json
{
  "direct": {
    "101": {
      "max_absolute_discrepancy": 1.9167433092661668,
      "mean_absolute_discrepancy": 0.14802584730264823,
      "median_absolute_discrepancy": 0.1176370408566072,
      "normalized_by_total_energy_range": 3.0080950440737567e-05,
      "p95_absolute_discrepancy": 0.354295939282656
    },
    "21": {
      "max_absolute_discrepancy": 38.45200032017965,
      "mean_absolute_discrepancy": 3.4723296259221077,
      "median_absolute_discrepancy": 2.6557262612514023,
      "normalized_by_total_energy_range": 0.0007456832747150523,
      "p95_absolute_discrepancy": 9.473414441334082
    },
    "convergence_pass": true
  },
  "dot": {
    "101": {
      "max_absolute_discrepancy": 1.914283470796363,
      "mean_absolute_discrepancy": 0.1691894411421481,
      "median_absolute_discrepancy": 0.14448144179505107,
      "normalized_by_total_energy_range": 3.149487695839892e-05,
      "p95_absolute_discrepancy": 0.37760868283646104
    },
    "21": {
      "max_absolute_discrepancy": 33.62902908573233,
      "mean_absolute_discrepancy": 4.151521385611986,
      "median_absolute_discrepancy": 3.515472197031272,
      "normalized_by_total_energy_range": 0.0007584426301676028,
      "p95_absolute_discrepancy": 10.104141011315791
    },
    "convergence_pass": true
  }
}
```

## Failures, exclusions, and limitations

Near-zero endpoint denominators excluded from only the normalized plot: {'none': 0, 'dot': 0, 'direct': 0}. They remain in the primary metric. NaN/tie/zero-field rates are retained in the CSV and per-trajectory parquet. Monotonicity is a landscape diagnostic; it does not establish generation or composition gains.

## Concise interpretation for Yilun

Checkpoint-only held-out evaluation gives **FAIL** under the preregistered epoch-8 paired-bootstrap rule. See the epoch-8 table above for absolute scores and paired confidence intervals; no checkpoint selection, retraining, guidance, sampler change, or weight modification was performed.
