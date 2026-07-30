# Fixed-candidate scalar-energy pilot

`t_eval=1.0` is the repository terminal/clean endpoint (`z_t=(1-t)noise+t*data`). Quality is independent DINO nearest-reference similarity plus supplied-label ImageNet probability.

|score|metric|estimate|95% CI|
|---|---|---:|---|
|direct_energy|spearman_quality_clustered|0.631|[0.592, 0.671]|
|direct_energy|pair_accuracy_clustered|0.715|[0.696, 0.732]|
|direct_energy|conditional_correct_lower|0.500|[0.438, 0.562]|
|direct_energy|corruption_increases_all_families|1.000|[1.000, 1.000]|
|direct_energy_zero_anchored|spearman_quality_clustered|0.633|[0.591, 0.674]|
|direct_energy_zero_anchored|pair_accuracy_clustered|0.715|[0.696, 0.736]|
|direct_energy_zero_anchored|conditional_correct_lower|0.516|[0.457, 0.578]|
|direct_energy_zero_anchored|corruption_increases_all_families|1.000|[1.000, 1.000]|
|dot_energy|spearman_quality_clustered|0.629|[0.590, 0.671]|
|dot_energy|pair_accuracy_clustered|0.714|[0.694, 0.732]|
|dot_energy|conditional_correct_lower|0.504|[0.441, 0.566]|
|dot_energy|corruption_increases_all_families|1.000|[1.000, 1.000]|
|dot_energy_zero_anchored|spearman_quality_clustered|0.629|[0.589, 0.668]|
|dot_energy_zero_anchored|pair_accuracy_clustered|0.714|[0.695, 0.732]|
|dot_energy_zero_anchored|conditional_correct_lower|0.504|[0.441, 0.566]|
|dot_energy_zero_anchored|corruption_increases_all_families|1.000|[1.000, 1.000]|
|base_field_norm|spearman_quality_clustered|0.613|[0.554, 0.670]|
|base_field_norm|pair_accuracy_clustered|0.711|[0.684, 0.737]|
|base_field_norm|conditional_correct_lower|0.453|[0.395, 0.516]|
|base_field_norm|corruption_increases_all_families|0.836|[0.773, 0.895]|
