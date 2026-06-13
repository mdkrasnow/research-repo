# Phase 1 — lever episode (3 seeds, 800 steps/stage)

Values are mean ± 95% CI. High neg_control_mse = honest (broken-symmetry task has no real symmetry to exploit). Low neg_control + high composition/identity/inverse error = shortcut cheating.

| Stage | clean_mse | neg_control_mse | shortcut_sens | comp_err | id_err | inv_err | verdicts |
|---|---|---|---|---|---|---|---|
| 1 pred-only (v0) | 0.009±0.038 | 0.168±0.569 | 0.919±0.321 | 6.896±15.137 | 1.986±4.442 | 2.337±1.827 | clean_win:1, shortcut_win:2 |
| 2 W-only | 0.008±0.036 | 0.310±1.180 | 1.056±0.188 | 6.214±11.111 | 1.900±3.998 | 1.798±1.434 | clean_win:1, shortcut_win:2 |
| 4 H->W | 0.019±0.050 | 1.014±0.043 | 0.007±0.021 | 0.008±0.006 | 0.379±0.514 | 0.334±0.332 | clean_win:3 |

**Gate (per-seed S4 neg_control > S2 neg_control): 3/3 pass**

## Statistical test — W-only vs H->W (Welch t, two-sided)

| metric | W-only mean | H->W mean | t | p | Cohen's d |
|---|---|---|---|---|---|
| composition_error | 6.214 | 0.0075 | 2.40 | 1.38e-01 | 1.96 |
| shortcut_sensitivity | 1.056 | 0.0070 | 23.86 | 1.54e-03 | 19.48 |
| identity_error | 1.900 | 0.3790 | 1.62 | 2.42e-01 | 1.33 |
| inverse_error | 1.798 | 0.3336 | 4.28 | 4.20e-02 | 3.50 |

Interpretation: W-only PRESERVES the shortcut (structural errors stay high, verdict stays shortcut_win); H->W REPAIRS it (shortcut_sensitivity and composition collapse to ~0; identity/inverse drop several-fold but not to 0; neg_control becomes honest). The Welch test quantifies the W-only -> H->W gap.
