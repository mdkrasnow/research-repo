# SIA-Lever-120B — policy comparison (measured eval-set regret)

Episodes: 9 (eval seeds, held-out). Regret = best cost-adjusted measured outcome − chosen action's measured outcome (real reruns, no transition table). W calls = actions that trigger a weight update (W or H_THEN_W).

| Policy | Lever Acc ↑ | Mean Regret ↓ | Max Regret ↓ | W calls | Invalid JSON |
|---|---|---|---|---|---|
| H_only | 0.33 | 0.371 | 0.912 | 0/9 | 0.00 |
| W_only | 0.33 | 0.443 | 1.022 | 9/9 | 0.00 |
| alternating | 0.44 | 0.297 | 0.832 | 4/9 | 0.00 |
| plateau_then_w | 0.67 | 0.000 | 0.000 | 6/9 | 0.00 |
| oracle_sandwich_rule | 1.00 | 0.000 | 0.000 | 6/9 | 0.00 |
| base_gpt_oss | 0.67 | 0.230 | 1.022 | 7/9 | 0.11 |
| gpt_oss_lora | 0.89 | 0.047 | 0.420 | 5/9 | 0.00 |
| oracle_best | 1.00 | 0.000 | 0.000 | 6/9 | 0.00 |

Notes:
- oracle_sandwich_rule is a deterministic upper-bound diagnostic (label-free rule).
- oracle_best is the unreachable ceiling (always the cost-adjusted best lever).
- base_gpt_oss / gpt_oss_lora appear only when rollout files are supplied (need endpoint/GPU).
- plateau_then_w is a paper-STYLE scheduler, not an exact SIA reproduction.
