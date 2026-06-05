# Phase 1 — lever episode (15 seeds, 2000 steps/stage)

Values are mean ± 95% CI. High neg_control_mse = honest (broken-symmetry task has no real symmetry to exploit). Low neg_control + high composition/identity/inverse error = shortcut cheating.

| Stage | clean_mse | neg_control_mse | shortcut_sens | comp_err | id_err | inv_err | verdicts |
|---|---|---|---|---|---|---|---|
| 1 pred-only (v0) | 0.003±0.004 | 0.116±0.128 | 0.975±0.039 | 5.059±1.554 | 1.713±0.483 | 3.056±1.553 | clean_win:2, shortcut_win:13 |
| 2 W-only | 0.003±0.004 | 0.145±0.172 | 1.009±0.048 | 4.658±1.414 | 1.665±0.479 | 2.732±1.436 | clean_win:2, shortcut_win:13 |
| 4 H->W | 0.006±0.004 | 1.067±0.014 | 0.002±0.001 | 0.001±0.000 | 0.324±0.079 | 0.309±0.075 | clean_win:15 |

**Gate (per-seed S4 neg_control > S2 neg_control): 15/15 pass**

## Statistical test — W-only vs H->W (Welch t, two-sided)

| metric | W-only mean | H->W mean | t | p | Cohen's d |
|---|---|---|---|---|---|
| composition_error | 4.658 | 0.0014 | 7.06 | 5.67e-06 | 2.58 |
| shortcut_sensitivity | 1.009 | 0.0015 | 44.60 | 1.63e-16 | 16.29 |
| identity_error | 1.665 | 0.3236 | 5.93 | 2.96e-05 | 2.17 |
| inverse_error | 2.732 | 0.3093 | 3.61 | 2.79e-03 | 1.32 |

Interpretation: W-only PRESERVES the shortcut (structural errors stay high, verdict stays shortcut_win); H->W REPAIRS it (all structural errors collapse to ~0, neg_control becomes honest). The Welch test quantifies the W-only -> H->W gap.
