# Rejection-payoff (label-enrichment)

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
