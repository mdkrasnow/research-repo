# Fixed-candidate scalar-energy pilot

`t_eval=1.0` is the repository terminal/clean endpoint (`z_t=(1-t)noise+t*data`). Quality is independent DINO nearest-reference similarity plus supplied-label ImageNet probability.

|score|metric|estimate|95% CI|
|---|---|---:|---|
|direct_energy|spearman_quality_clustered|0.177|[0.065, 0.281]|
|direct_energy|pair_accuracy_clustered|0.557|[0.523, 0.590]|
|direct_energy|conditional_correct_lower|0.527|[0.469, 0.586]|
|direct_energy|corruption_increases_all_families|1.000|[1.000, 1.000]|
|direct_energy_zero_anchored|spearman_quality_clustered|0.190|[0.084, 0.292]|
|direct_energy_zero_anchored|pair_accuracy_clustered|0.562|[0.528, 0.596]|
|direct_energy_zero_anchored|conditional_correct_lower|0.527|[0.469, 0.586]|
|direct_energy_zero_anchored|corruption_increases_all_families|1.000|[1.000, 1.000]|
|dot_energy|spearman_quality_clustered|0.185|[0.077, 0.284]|
|dot_energy|pair_accuracy_clustered|0.560|[0.527, 0.590]|
|dot_energy|conditional_correct_lower|0.484|[0.426, 0.543]|
|dot_energy|corruption_increases_all_families|1.000|[1.000, 1.000]|
|dot_energy_zero_anchored|spearman_quality_clustered|0.185|[0.072, 0.288]|
|dot_energy_zero_anchored|pair_accuracy_clustered|0.560|[0.527, 0.591]|
|dot_energy_zero_anchored|conditional_correct_lower|0.484|[0.422, 0.547]|
|dot_energy_zero_anchored|corruption_increases_all_families|1.000|[1.000, 1.000]|
|base_field_norm|spearman_quality_clustered|0.295|[0.173, 0.405]|
|base_field_norm|pair_accuracy_clustered|0.599|[0.557, 0.636]|
|base_field_norm|conditional_correct_lower|0.527|[0.465, 0.586]|
|base_field_norm|corruption_increases_all_families|0.793|[0.719, 0.859]|
