# Baseline comparison table

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
