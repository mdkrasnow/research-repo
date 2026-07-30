# Fixed-candidate scalar-energy pilot

`t_eval=1.0` is the repository terminal/clean endpoint (`z_t=(1-t)noise+t*data`). Quality is independent DINO nearest-reference similarity plus supplied-label ImageNet probability.

|score|metric|estimate|95% CI|
|---|---|---:|---|
|direct_energy|spearman_quality_clustered|0.783|[0.751, 0.817]|
|direct_energy|pair_accuracy_clustered|0.786|[0.769, 0.804]|
|direct_energy|conditional_correct_lower|0.527|[0.465, 0.590]|
|direct_energy|corruption_increases_all_families|1.000|[1.000, 1.000]|
|direct_energy_zero_anchored|spearman_quality_clustered|0.782|[0.748, 0.816]|
|direct_energy_zero_anchored|pair_accuracy_clustered|0.784|[0.767, 0.802]|
|direct_energy_zero_anchored|conditional_correct_lower|0.480|[0.418, 0.539]|
|direct_energy_zero_anchored|corruption_increases_all_families|1.000|[1.000, 1.000]|
|dot_energy|spearman_quality_clustered|0.779|[0.746, 0.812]|
|dot_energy|pair_accuracy_clustered|0.783|[0.768, 0.800]|
|dot_energy|conditional_correct_lower|0.461|[0.398, 0.523]|
|dot_energy|corruption_increases_all_families|1.000|[1.000, 1.000]|
|dot_energy_zero_anchored|spearman_quality_clustered|0.779|[0.747, 0.814]|
|dot_energy_zero_anchored|pair_accuracy_clustered|0.783|[0.767, 0.800]|
|dot_energy_zero_anchored|conditional_correct_lower|0.461|[0.398, 0.523]|
|dot_energy_zero_anchored|corruption_increases_all_families|1.000|[1.000, 1.000]|
|base_field_norm|spearman_quality_clustered|0.634|[0.582, 0.684]|
|base_field_norm|pair_accuracy_clustered|0.718|[0.696, 0.741]|
|base_field_norm|conditional_correct_lower|0.512|[0.453, 0.574]|
|base_field_norm|corruption_increases_all_families|0.871|[0.809, 0.926]|
