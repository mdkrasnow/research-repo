# Rejection-payoff (label-enrichment)

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
