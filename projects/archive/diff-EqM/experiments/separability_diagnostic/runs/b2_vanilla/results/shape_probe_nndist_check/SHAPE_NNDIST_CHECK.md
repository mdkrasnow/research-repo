# SHAPE probe -- nn_dist skepticism control

held-out n=450 (70/30 split, seed=4)

1. raw shape-probe AUROC (held-out): 0.8271
2. Pearson corr(p_bad, nn_dist): 0.5377
   Spearman corr(p_bad, nn_dist): 0.5267
3. AUROC after residualizing p_bad against nn_dist: 0.5627
4. within-nn_dist-decile AUROC: 0.6086 (1/10 usable bins)

## VERDICT: COLLAPSE: residual AUROC 0.563 falls toward the baseline floor, same failure mode as endpoint E_psi. DOWNGRADE the whole claim to 'early trajectory predicts nn_dist-defined failure', not 'trajectory dynamics reveal semantic OOD'.
