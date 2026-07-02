# EqM Separability — Final Readout

Run dir: `runs/b2_vanilla`

## Verdict
```
EqM SEPARABILITY DIAGNOSTIC -- VERDICT
==================================================
samples: good=750 garbage=750
label sanity (s4 latent-NN raw AUROC, should be >~0.6): 0.592
  !! s4 weak -> LABEL PIPELINE SUSPECT; treat all below with caution.

de-confounded (within-norm-bin) AUROC, fixed regime:
  s1: raw=0.505  within-norm=0.609   dot energy  -<f,x>  (de-confoundable)
  s2: raw=0.548  within-norm=0.539   l2 energy  0.5||f||^2  (norm-coupled)
  s3: raw=0.568  within-norm=0.605   path integral  sum<f,dx>  (de-confoundable)
  s4: raw=0.592  within-norm=0.627   latent NN dist  (no f; label sanity)
  s5: raw=0.548  within-norm=0.539   post-step residual ||f||  (norm-coupled)

best_independent_auroc = 0.609  (score s1)

VERDICT: WEAK: separation exists but below the 0.80 action threshold (best 0.609). Quantified target: lift s1 >= 0.80. This is how much a 'more reliable scalar' must improve before the metacognition sampler is worth building.
```

## Decision: **GREEN**
Best de-confounded within-norm AUROC: 0.819 (learned trajectory probe (PROBE_VERDICT.txt)).

**Next action:** Run probe-guided rejection sampler: `fid_payoff.py` (pool reject) then `probe_gated_sample.py` / `online_adaptive_sampler.py` (in-line restart).

## Baseline table
Run: `runs/b2_vanilla`  n_good=750 n_garbage=750  seed=0  verdict on within-norm AUROC (de-confounded): GREEN≥0.80 / WEAK 0.60–0.80 / KILL<0.60.

| method | raw AUROC | within-norm AUROC | bins | verdict |
|---|---|---|---|---|
| random | 0.500 | 0.500 | — | KILL |
| ||f|| only (norm) | 0.548 | 0.538 | 5 | KILL |
| s1: -<f,x>  (dot energy) | 0.505 | 0.609 | 5 | WEAK |
| s3: sum<f,dx>  (path integral) | 0.568 | 0.605 | 5 | WEAK |
| s2: 0.5||f||^2  (l2 energy, norm-coupled) | 0.548 | 0.538 | 5 | KILL |
| s5: post-step ||f||  (residual, norm-coupled) | 0.548 | 0.538 | 5 | KILL |
| latent-NN s4 (no f; sanity) | 0.592 | 0.627 | 5 | WEAK |
| learned trajectory probe (PROBE_VERDICT.txt) | — | 0.819 | 3 | GREEN |


## Rejection-payoff
Run: `runs/b2_vanilla`  seed=0  n_good=750 n_garb=750  base_good_frac=0.500

Payoff = good-fraction(kept) − base. Higher = score concentrates garbage in the rejected tail.
Controls: random (floor), norm_only (confounder), shuf_score/shuf_label (must ≈0).

| method | enr@10 | enr@20 | enr@30 | enr@40 | mean_nn@40 |
|---|---|---|---|---|---|
| s1 | +0.007 | +0.012 | +0.008 | +0.006 | 14.534 |
| s2 | +0.014 | +0.019 | +0.023 | +0.023 | 14.375 |
| s3 | +0.019 | +0.027 | +0.025 | +0.038 | 14.216 |
| s5 | +0.014 | +0.019 | +0.023 | +0.023 | 14.375 |
| norm_only | -0.006 | -0.011 | -0.012 | -0.022 | 14.761 |
| latent_nn_s4 | +0.019 | +0.026 | +0.034 | +0.052 | 14.141 |
| random | +0.001 | -0.003 | -0.003 | +0.001 | 14.493 |
| shuf_score[s3] | -0.004 | -0.013 | -0.013 | -0.014 | 14.745 |
| shuf_label[s3] | +0.002 | +0.005 | +0.012 | +0.023 | 14.634 |

**Best real score @30%:** s3 enr=+0.025 (random -0.003, norm_only -0.012, shuf_label +0.012). Gain over random = +0.028.

## FID-payoff hook
Label-enrichment is the cheap payoff. For FID payoff (rank→keep→FID vs random/oracle), run `fid_payoff.py` (GPU; Inception). This script leaves that to the GPU path.


## Controls present
- rejection: random, norm_only, shuffled-score, shuffled-label
- AUROC: raw vs within-norm (matched-norm de-confound), latent-NN s4 sanity
- probe (if run): held-out split, fixed seed, label-shuffle

