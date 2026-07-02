# Post-hoc energy/OOD head — results

good=750 garbage=750 ambiguous=1500
NOTE: knn_dist/nn_dist excluded as a baseline -- labels are derived by thresholding nn_dist itself (thresholds.json), so it is circular (AUROC=1.0 by construction), not a real comparison point.
mined hard negatives: 1125 (fraction that are true-labeled garbage: 0.407, rest ambiguous — near-manifold pool)
E_psi: 5-seed mean 0.7525 +/- 0.0094

| method | AUROC |
|---|---|
| endpoint_dot (dead energy) | 0.5053 |
| path_integral_dot (dead energy) | 0.5678 |
| final_norm_magnitude | 0.5480 |
| E_psi post-hoc energy head (5 seeds) | 0.7525 |
| SHAPE-only descent probe (reference ceiling) | 0.8152 |

## E_psi vs nn_dist (held-out test split only, is E_psi just a distance proxy?)
- Pearson corr(E_psi, nn_dist): 0.431
- Spearman corr(E_psi, nn_dist): 0.430
- AUROC of E_psi residual after regressing out nn_dist: 0.5242 +/- 0.0128 (vs raw E_psi AUROC 0.7525)

## VERDICT: PARTIAL: E_psi (0.753) beats old energy baselines (best 0.568) but well short of SHAPE probe (0.815). Some endpoint-only signal recoverable via hard-negative mining, but most of the metacog signal is still in trajectory shape. nn_dist check: moderate corr (Pearson 0.431, Spearman 0.430) -- E_psi recovers a mix of distance-correlated and distance-independent signal. Residual-after-nn_dist AUROC 0.524 is the honest estimate of signal beyond distance.
