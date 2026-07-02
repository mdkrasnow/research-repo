# EqM Separability — Final Readout

Run dir: `/n/home03/mkrasnow/research-repo/projects/diff-EqM/experiments/separability_diagnostic/runs/b2_v10_anm`

## Verdict
```
EqM SEPARABILITY DIAGNOSTIC -- VERDICT
==================================================
samples: good=205 garbage=154
label sanity (s4 latent-NN raw AUROC, should be >~0.6): 0.583
  !! s4 weak -> LABEL PIPELINE SUSPECT; treat all below with caution.

de-confounded (within-norm-bin) AUROC, fixed regime:
  s1: raw=0.511  within-norm=0.592   dot energy  -<f,x>  (de-confoundable)
  s2: raw=0.520  within-norm=0.540   l2 energy  0.5||f||^2  (norm-coupled)
  s3: raw=0.538  within-norm=0.553   path integral  sum<f,dx>  (de-confoundable)
  s4: raw=0.583  within-norm=0.621   latent NN dist  (no f; label sanity)
  s5: raw=0.520  within-norm=0.540   post-step residual ||f||  (norm-coupled)
  s6: raw=0.542  within-norm=0.561   traj mean norm  (dynamics; norm-coupled)
  s7: raw=0.551  within-norm=0.540   log-norm decay slope  (dynamics; de-confoundable)
  s8: raw=0.614  within-norm=0.622   norm oscillation  (dynamics; de-confoundable)
  s9: raw=0.510  within-norm=0.539   late-stage slope  (dynamics; de-confoundable)

best_independent_auroc = 0.622  (score s8)

VERDICT: WEAK: separation exists but below the 0.80 action threshold (best 0.622). Quantified target: lift s8 >= 0.80. This is how much a 'more reliable scalar' must improve before the metacognition sampler is worth building.
```

## Decision: **WEAK**
Best de-confounded within-norm AUROC: 0.700 (learned trajectory probe (dynamics_probe/probe_table.csv)).

**Next action:** Run the learned trajectory probe: `dynamics_probe.py` + `learned_probe.py` (full descent-shape features, held-out, shuffle control).

## Baseline table
Run: `/n/home03/mkrasnow/research-repo/projects/diff-EqM/experiments/separability_diagnostic/runs/b2_v10_anm`  n_good=179 n_garbage=180  seed=0  verdict on within-norm AUROC (de-confounded): GREEN≥0.80 / WEAK 0.60–0.80 / KILL<0.60.

| method | raw AUROC | within-norm AUROC | bins | verdict |
|---|---|---|---|---|
| random | 0.500 | 0.500 | — | KILL |
| ||f|| only (norm) | 0.520 | 0.540 | 5 | KILL |
| s1: -<f,x>  (dot energy) | 0.511 | 0.592 | 5 | KILL |
| s3: sum<f,dx>  (path integral) | 0.537 | 0.553 | 5 | KILL |
| s2: 0.5||f||^2  (l2 energy, norm-coupled) | 0.520 | 0.540 | 5 | KILL |
| s5: post-step ||f||  (residual, norm-coupled) | 0.520 | 0.540 | 5 | KILL |
| latent-NN s4 (no f; sanity) | 0.583 | 0.621 | 5 | WEAK |
| learned trajectory probe (dynamics_probe/probe_table.csv) | — | 0.700 | 3 | WEAK |


## Rejection-payoff
Run: `/n/home03/mkrasnow/research-repo/projects/diff-EqM/experiments/separability_diagnostic/runs/b2_v10_anm`  seed=0  n_good=205 n_garb=154  base_good_frac=0.571

Payoff = good-fraction(kept) − base. Higher = score concentrates garbage in the rejected tail.
Controls: random (floor), norm_only (confounder), shuf_score/shuf_label (must ≈0).

| method | enr@10 | enr@20 | enr@30 | enr@40 | mean_nn@40 |
|---|---|---|---|---|---|
| s1 | +0.008 | +0.018 | +0.011 | +0.010 | 15.690 |
| s2 | +0.008 | +0.014 | +0.011 | +0.024 | 15.739 |
| s3 | -0.001 | +0.011 | +0.019 | +0.034 | 15.533 |
| s5 | +0.008 | +0.014 | +0.011 | +0.024 | 15.739 |
| s6 | +0.002 | +0.007 | +0.027 | +0.029 | 15.539 |
| s7 | -0.001 | +0.021 | +0.011 | +0.024 | 15.540 |
| s8 | +0.023 | +0.056 | +0.058 | +0.071 | 15.251 |
| s9 | -0.001 | +0.000 | +0.011 | +0.020 | 15.692 |
| norm_only | +0.005 | +0.000 | -0.005 | -0.013 | 15.856 |
| latent_nn_s4 | +0.033 | +0.035 | +0.031 | +0.024 | 15.530 |
| random | +0.008 | +0.000 | -0.001 | +0.006 | 15.771 |
| shuf_score[s8] | -0.001 | -0.010 | -0.001 | -0.008 | 15.861 |
| shuf_label[s8] | +0.020 | +0.025 | +0.015 | +0.024 | 15.830 |

**Best real score @30%:** s8 enr=+0.058 (random -0.001, norm_only -0.005, shuf_label +0.015). Gain over random = +0.060.

## FID-payoff hook
Label-enrichment is the cheap payoff. For FID payoff (rank→keep→FID vs random/oracle), run `fid_payoff.py` (GPU; Inception). This script leaves that to the GPU path.


## Controls present
- rejection: random, norm_only, shuffled-score, shuffled-label
- AUROC: raw vs within-norm (matched-norm de-confound), latent-NN s4 sanity
- probe (if run): held-out split, fixed seed, label-shuffle

