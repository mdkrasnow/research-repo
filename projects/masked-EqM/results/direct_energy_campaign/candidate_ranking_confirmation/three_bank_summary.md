# Three-bank direct-energy confirmation

All rows use raw scalar energy. Positive comparison values mean direct is better under the lower-is-better convention. CIs resample the three complete, fixed candidate banks and are correspondingly coarse.

| metric | direct mean [bank CI] | dot mean [bank CI] | base mean [bank CI] | direct-dot [bank CI] | direct wins |
|---|---:|---:|---:|---:|---:|
| spearman_quality_clustered | 0.530 [0.177, 0.783] | 0.531 [0.185, 0.779] | 0.514 [0.295, 0.634] | -0.001 [-0.008, 0.004] | 2/3 |
| pair_accuracy_clustered | 0.686 [0.557, 0.786] | 0.685 [0.560, 0.783] | 0.676 [0.599, 0.718] | 0.000 [-0.003, 0.003] | 2/3 |
| conditional_correct_lower | 0.518 [0.500, 0.527] | 0.483 [0.461, 0.504] | 0.497 [0.453, 0.527] | 0.035 [-0.004, 0.066] | 2/3 |
| corruption_increases_all_families | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.833 [0.793, 0.871] | 0.000 [0.000, 0.000] | 0/3 |
