# Phase 3 — lever attribution prototype (10 seeds x 3 modes = 30 episodes)

Three-mode prototype (not a large benchmark). Regret = best-achievable true-score (over levers, REAL reruns) - chosen lever's true-score. RAW scores are measured once; the W_COST sweep below re-scores the SAME raw numbers (no transition table, no extra training).

## Headline (W_COST = 0.05)

| Policy | lever accuracy | mean regret | max regret |
|---|---|---|---|
| H_only | 0.33 (10/30) | 0.318 | 0.913 |
| W_only | 0.33 (10/30) | 0.421 | 1.023 |
| alternating | 0.67 (20/30) | 0.148 | 0.832 |
| selector | 0.97 (29/30) | 0.014 | 0.360 |

Wilcoxon (selector regret < alternating, paired): W=8.0, p=1.312e-02
Oracle best-lever matches pre-registered correct lever on 29/30 episodes.

## W_COST sensitivity (mean regret; selector should stay lowest across the range)

| W_COST | H_only | W_only | alternating | selector | oracle match |
|---|---|---|---|---|---|
| 0.00 | 0.355 | 0.408 | 0.168 | 0.019 | 26/30 |
| 0.01 | 0.346 | 0.409 | 0.163 | 0.017 | 28/30 |
| 0.03 | 0.331 | 0.414 | 0.155 | 0.015 | 28/30 |
| 0.05 | 0.318 | 0.421 | 0.148 | 0.014 | 29/30 |
| 0.10 | 0.287 | 0.440 | 0.134 | 0.015 | 28/30 |

Selector achieves the lowest mean regret at every W_COST in [0.00, 0.10] — the result does not depend on the particular cost chosen. At W_COST=0 (pure quality, no cost preference) the bad_verifier mode's H and H_THEN_W tie; the W_COST only expresses a transparent preference for the cheaper lever when both reach the same quality.
