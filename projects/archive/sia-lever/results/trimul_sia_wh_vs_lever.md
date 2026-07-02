# SIA-W+H (paper-style) vs SIA-Lever — TriMul GPU-kernel task

Eval episodes: 96 (cache: `gpt_oss/data/out/kernel_cache.jsonl`)

| policy | lever_acc | mean_regret | max_regret | w_calls | invalid_json |
|---|---|---|---|---|---|
| oracle_best   [POS control] | 1.000 | 0.0000 | 0.0000 | 72 | 0.000 |
| W_only        [NEG control] | 0.000 | 1.0060 | 1.0435 | 96 | 0.000 |
| H_only        [NEG control] | 0.250 | 0.7076 | 0.9435 | 0 | 0.000 |
| sia_wh_plateau [paper-style] | 1.000 | 0.0000 | 0.0000 | 72 | 0.000 |
| sia_lever_rule [ours] | 0.750 | 0.2359 | 0.9435 | 48 | 0.000 |

**Read between the controls.** oracle_best = upper bound (regret 0); W_only / H_only = floors. A treatment near the floor = dead; near oracle = works.

**Honesty:** the public SIA repo ships the harness (H) loop only — no W code — so `sia_wh_plateau` is a paper-STYLE reconstruction of the paper's H-vs-W scheduler, not an exact reproduction. SIA-Lever's edge is naming the **minimal correct lever** (lever_acc + fewer paid W retrains at equal regret), not necessarily lower regret than a competent scheduler. See documentation/reproduction_limits.md.
