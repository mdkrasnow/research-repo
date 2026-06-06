# SIA-Lever-120B — DEMO REPORT

One place for the demo: the phenomenon, the lever-attribution result, and the gpt-oss-120b base-vs-LoRA comparison. Figures are real CPU measurements unless marked PREVIEW (synthetic, replaced by the GPU run) or pending.

Refresh: `python3 scripts/make_demo_report.py` (after `run_gpu_comparison.sh` for real gpt-oss numbers).

---

## 1. The phenomenon (CPU, measured, 15 seeds)
W-only PRESERVES shortcut-cheating; H→W REPAIRS it. Welch t: shortcut_sens d=16.3 (p=1.6e-16), composition d=2.58. Gate 15/15.

![Phase 1: W-only preserves the shortcut; H→W learns the real group](results/episode_plot.png)
*Phase 1: W-only preserves the shortcut; H→W learns the real group* — `results/episode_plot.png`

![Phase 3: lever-selector lowest regret across the whole W_COST sweep](results/phase3_plot.png)
*Phase 3: lever-selector lowest regret across the whole W_COST sweep* — `results/phase3_plot.png`

## 2. Lever-attribution policy comparison (measured regret)

# SIA-Lever-120B — policy comparison (measured eval-set regret)

Episodes: 9 (eval seeds, held-out). Regret = best cost-adjusted measured outcome − chosen action's measured outcome (real reruns, no transition table). W calls = actions that trigger a weight update (W or H_THEN_W).

| Policy | Lever Acc ↑ | Mean Regret ↓ | Max Regret ↓ | W calls | Invalid JSON |
|---|---|---|---|---|---|
| H_only | 0.33 | 0.371 | 0.912 | 0/9 | 0.00 |
| W_only | 0.33 | 0.443 | 1.022 | 9/9 | 0.00 |
| alternating | 0.44 | 0.297 | 0.832 | 4/9 | 0.00 |
| plateau_then_w | 0.67 | 0.000 | 0.000 | 6/9 | 0.00 |
| oracle_sandwich_rule | 1.00 | 0.000 | 0.000 | 6/9 | 0.00 |
| oracle_best | 1.00 | 0.000 | 0.000 | 6/9 | 0.00 |

![Regret / lever-accuracy / W-call-rate by policy (green=oracle/rule, gray=baseline)](plots/final_comparison.png)
*Regret / lever-accuracy / W-call-rate by policy (green=oracle/rule, gray=baseline)* — `plots/final_comparison.png`

## 3. gpt-oss-120b base vs LoRA selector (PREVIEW — synthetic)

> **PREVIEW**: synthetic base+LoRA rollouts illustrate the figures. The GPU run (`run_gpu_comparison.sh`) overwrites these with real gpt-oss-120b measurements.

# Adapter eval — PREVIEW (20260606T034130Z)

Figure: `adapter_eval_20260606T034130Z.png` (base vs +PREVIEW)

| metric | base gpt-oss | gpt-oss+LoRA | delta |
|---|---|---|---|
| lever_accuracy | 0.667 | 0.889 | +0.222 |
| mean_regret | 0.230 | 0.047 | -0.183 |
| max_regret | 1.022 | 0.420 | -0.601 |
| invalid_json_rate | 0.111 | 0.000 | -0.111 |

Adapter per-mode accuracy: {"shortcut_leak": 1.0, "model_prior_gap": 0.6666666666666666, "bad_verifier": 1.0}

### What LoRA changed vs base

vs base
- **fixed** (base wrong → LoRA right): 3
- regressed (base right → LoRA wrong): 1
- both right: 5 · both wrong: 0
  - [fixed] model_prior_gap_seed_007 (model_prior_gap): base=`None` lora=`W` correct=`W`
  - [fixed] bad_verifier_seed_007 (bad_verifier): base=`W` lora=`H` correct=`H`
  - [fixed] bad_verifier_seed_009 (bad_verifier): base=`W` lora=`H` correct=`H`
  - [regressed] model_prior_gap_seed_008 (model_prior_gap): base=`W` lora=`H` correct=`W`

![Base vs +LoRA: lever accuracy ↑, mean regret ↓, invalid-JSON ↓](results/gpt_oss/preview/adapter_eval_20260606T034130Z.png)
*Base vs +LoRA: lever accuracy ↑, mean regret ↓, invalid-JSON ↓* — `results/gpt_oss/preview/adapter_eval_20260606T034130Z.png`

![Per-mode lever accuracy: base vs LoRA](results/gpt_oss/preview/adapter_per_mode_20260606T034130Z.png)
*Per-mode lever accuracy: base vs LoRA* — `results/gpt_oss/preview/adapter_per_mode_20260606T034130Z.png`

## 4. LoRA training curve

_(pending) SFT/DPO loss (and GRPO reward) vs step_

## 5. Official SIA-H + LawBench (stretch)

_(pending) Official SIA-H: lever accuracy / regret vs generation_

_(pending) LawBench: ours vs paper (13.5 / 50.0 / 70.1; * = reduced split)_

---

## Diagnostics (for debugging a run)
- per-episode mistakes + raw model output: `results/gpt_oss/<tag>_diagnostics.md`
- base→LoRA fixed/regressed episode diff: in the adapter_eval `.md`
- action distribution + per-mode accuracy: `<tag>_action_dist.png`, `<tag>_per_mode.png`
- adapter provenance (hashes/gpu/config): `adapters/gpt_oss_120b/<run>/`
- env/endpoint readiness: `python3 gpt_oss/check_env.py`

