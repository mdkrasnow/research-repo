# v17 MorphismGym — Results Report

## Phase 1 — Discovery

### decoy_pressure (true=['rotate', 'hue']) — PASS
| arm | cov | | arm | cov |
|---|---|---|---|---|
| BASE | 0.000 | | ORACLE | 0.975 |
| RANDOM_VALID | 0.935 | | LEARNED_SINGLE | 0.458 |
| **LEARNED_MULTI** | **0.782** | | NO_ANCHOR | 0.766 |
| NO_DIVERSITY | 0.558 | | | |
- learned: true_use=0.67 decoy_use=0.00 recall=1.00 | NO_ANCHOR decoy_use=0.00 (ablation should rise) | gate: decoy_pressure: recall>=.5, STRONG decoy-avoid(<.10), valid>decoy-arm, NO_ANCHOR validity worse

### impossible_control (true=[]) — PASS
| arm | cov | | arm | cov |
|---|---|---|---|---|
| BASE | 0.000 | | ORACLE | 0.000 |
| RANDOM_VALID | 0.000 | | LEARNED_SINGLE | 0.000 |
| **LEARNED_MULTI** | **0.000** | | NO_ANCHOR | 0.000 |
| NO_DIVERSITY | 0.000 | | | |
- learned: true_use=0.00 decoy_use=0.00 recall=0.00 | NO_ANCHOR decoy_use=0.00 (ablation should rise) | gate: no-hallucination (cov<0.15)

### multi_composed (true=['rotate', 'scale', 'translate_x']) — PASS
| arm | cov | | arm | cov |
|---|---|---|---|---|
| BASE | 0.000 | | ORACLE | 1.000 |
| RANDOM_VALID | 1.000 | | LEARNED_SINGLE | 0.391 |
| **LEARNED_MULTI** | **0.879** | | NO_ANCHOR | 0.985 |
| NO_DIVERSITY | 0.519 | | | |
- learned: true_use=0.63 decoy_use=0.00 recall=0.78 | NO_ANCHOR decoy_use=0.00 (ablation should rise) | gate: multi: recall>=.66, low-decoy, cov>single, valid>decoy-arm, NO_ANCHOR validity worse

### multi_independent (true=['rotate', 'hue', 'scale']) — PASS
| arm | cov | | arm | cov |
|---|---|---|---|---|
| BASE | 0.000 | | ORACLE | 0.983 |
| RANDOM_VALID | 0.957 | | LEARNED_SINGLE | 0.330 |
| **LEARNED_MULTI** | **0.969** | | NO_ANCHOR | 0.844 |
| NO_DIVERSITY | 0.562 | | | |
- learned: true_use=0.69 decoy_use=0.01 recall=0.89 | NO_ANCHOR decoy_use=0.00 (ablation should rise) | gate: multi: recall>=.66, low-decoy, cov>single, valid>decoy-arm, NO_ANCHOR validity worse

### single_hue (true=['hue']) — PASS
| arm | cov | | arm | cov |
|---|---|---|---|---|
| BASE | 0.000 | | ORACLE | 0.953 |
| RANDOM_VALID | 0.870 | | LEARNED_SINGLE | 0.929 |
| **LEARNED_MULTI** | **0.918** | | NO_ANCHOR | 0.566 |
| NO_DIVERSITY | 0.917 | | | |
- learned: true_use=0.56 decoy_use=0.00 recall=1.00 | NO_ANCHOR decoy_use=0.00 (ablation should rise) | gate: single: true>decoy usage, low-decoy, valid>decoy-arm, NO_ANCHOR validity worse

### single_rotation (true=['rotate']) — PASS
| arm | cov | | arm | cov |
|---|---|---|---|---|
| BASE | 0.000 | | ORACLE | 1.000 |
| RANDOM_VALID | 1.000 | | LEARNED_SINGLE | 1.000 |
| **LEARNED_MULTI** | **1.000** | | NO_ANCHOR | 0.954 |
| NO_DIVERSITY | 1.000 | | | |
- learned: true_use=0.89 decoy_use=0.00 recall=1.00 | NO_ANCHOR decoy_use=0.00 (ablation should rise) | gate: single: true>decoy usage, low-decoy, valid>decoy-arm, NO_ANCHOR validity worse

### single_scale (true=['scale']) — PASS
| arm | cov | | arm | cov |
|---|---|---|---|---|
| BASE | 0.000 | | ORACLE | 1.000 |
| RANDOM_VALID | 1.000 | | LEARNED_SINGLE | 0.070 |
| **LEARNED_MULTI** | **1.000** | | NO_ANCHOR | 1.000 |
| NO_DIVERSITY | 0.070 | | | |
- learned: true_use=0.08 decoy_use=0.00 recall=1.00 | NO_ANCHOR decoy_use=0.00 (ablation should rise) | gate: single: true>decoy usage, low-decoy, valid>decoy-arm, NO_ANCHOR validity worse

**Phase 1 fan-in: PASS** {'decoy_pressure': True, 'impossible_control': True, 'multi_composed': True, 'multi_independent': True, 'single_hue': True, 'single_rotation': True, 'single_scale': True}
