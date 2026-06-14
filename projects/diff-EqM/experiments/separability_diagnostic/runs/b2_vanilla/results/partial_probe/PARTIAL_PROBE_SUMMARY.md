EqM PARTIAL-trajectory probe — how early is failure detectable?
============================================================
good=750 garbage=750 full_T=249
k_frac  k_dec  dim   raw_AUROC        within-norm (de-conf)
 0.40   100   30   0.813±0.012     0.814±0.013
 0.50   124   30   0.814±0.011     0.814±0.011
 0.60   149   30   0.815±0.012     0.812±0.015
 0.70   174   30   0.817±0.011     0.818±0.014
 0.80   199   30   0.819±0.010     0.819±0.011
 0.90   224   30   0.819±0.009     0.819±0.011
 1.00   249   30   0.817±0.010     0.818±0.012

action bar (de-conf within-norm) = 0.75
deploy k_dec = 100 (artifact partial_probe_k100.npz)

## VERDICT: ONLINE-VIABLE: de-confounded AUROC >= 0.75 as early as k_frac=0.40 (step 100/249) -> enough early signal to build an equal-NFE online adaptive sampler.
