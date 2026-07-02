# Baseline comparison table

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
