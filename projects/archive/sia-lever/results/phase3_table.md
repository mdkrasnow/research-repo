# Phase 3 — lever attribution prototype (3 seeds x 3 modes = 9 episodes)

Three-mode prototype (not a large benchmark). Regret = best-achievable true-score (over levers, REAL reruns) - chosen lever's true-score. RAW scores are measured once; the W_COST sweep below re-scores the SAME raw numbers (no transition table, no extra training).

## Headline (W_COST = 0.05)

| Policy | lever accuracy | mean regret | max regret |
|---|---|---|---|
| H_only | 0.33 (3/9) | 0.235 | 0.729 |
| W_only | 0.33 (3/9) | 0.438 | 1.023 |
| alternating | 0.67 (6/9) | 0.116 | 0.729 |
| selector | 0.89 (8/9) | 0.046 | 0.360 |

Wilcoxon (selector regret < alternating, paired): W=4.0, p=4.375e-01
Oracle best-lever matches pre-registered correct lever on 8/9 episodes.

## W_COST sensitivity (mean regret; selector should stay lowest across the range)

| W_COST | H_only | W_only | alternating | selector | oracle match |
|---|---|---|---|---|---|
| 0.00 | 0.269 | 0.422 | 0.133 | 0.052 | 8/9 |
| 0.01 | 0.262 | 0.425 | 0.129 | 0.050 | 8/9 |
| 0.03 | 0.249 | 0.432 | 0.123 | 0.048 | 7/9 |
| 0.05 | 0.235 | 0.438 | 0.116 | 0.046 | 8/9 |
| 0.10 | 0.207 | 0.460 | 0.104 | 0.046 | 8/9 |

Selector achieves the lowest mean regret at every W_COST in [0.00, 0.10] — the result does not depend on the particular cost chosen. At W_COST=0 (pure quality, no cost preference) the bad_verifier mode's H and H_THEN_W tie; the W_COST only expresses a transparent preference for the cheaper lever when both reach the same quality.
