# Results Summary — SIA-Lever

## Research question
When a self-improving agent's run fails, should it update the harness (H) or the weights (W)?
Demonstrate one real case where the lever choice is decisive.

## Core claim
W-only updates preserve shortcut-cheating; H-first-then-W (fix verifier, then retrain) repairs it.

## Key findings (MEASURED, 15 seeds, 2000 steps/stage)
Toy = **adversarial shortcut trap** (shortcut leaks full target), not a naturalistic benchmark.
Metric suite upgraded: ill-posed equivariance proxy replaced by group-axiom errors (composition,
identity, inverse). Defensible wording: W-only **preserves** the cheat; H→W **repairs** it.

- **Phenomenon (15 seeds).** Prediction-only learner exploits the shortcut: clean_mse ~0 but solves
  the broken-symmetry control (neg_control low) and violates group axioms (composition 5.06,
  identity 1.71, inverse 3.06). `shortcut_win` 13/15.
- **W-only PRESERVES it.** More prediction-reward training leaves structural errors ~unchanged
  (shortcut_sens 0.98→1.01, composition 5.06→4.66). Still `shortcut_win` 13/15.
- **H→W REPAIRS it (15/15).** neg_control 0.15→1.067 (honest), shortcut_sens →0.002, composition
  →0.001, identity 1.67→0.32, inverse 2.73→0.31. `clean_win` 15/15.
- **Gate: 15/15** S4 neg_control > S2 neg_control.

### Statistical test — Welch t, W-only vs H→W (`results/episode_table.md`)
| metric | W-only | H→W | t | p | Cohen's d |
|---|---|---|---|---|---|
| shortcut_sensitivity | 1.009 | 0.002 | 44.6 | 1.6e-16 | 16.3 |
| composition_error | 4.658 | 0.001 | 7.06 | 5.7e-6 | 2.58 |
| identity_error | 1.665 | 0.324 | 5.93 | 3.0e-5 | 2.17 |
| inverse_error | 2.732 | 0.309 | 3.61 | 2.8e-3 | 1.32 |

All contrasts significant, large effect sizes. Plot: `results/episode_plot.png`.

Honest caveat: 2/15 seeds the prediction-only model partially learned rotation (verdict clean_win)
rather than fully cheating; the H→W repair is unanimous (15/15) and tight (low CI).

### Leak-strength robustness — two findings (`results/leak_sweep_table.md`, `leak_sweep_plot.png`)
1. **Shortcut-cheating is adversarial:** dominates ONLY at full leak α=1.0 (shortcut_win 0.80,
   neg_control 0.17). At α≤0.75 prediction-only stops reading the noisy leak (shortcut_sens→0,
   neg_control→honest ~1.0). Confirms Phase 1 is an adversarial trap, not a universal claim.
2. **H→W repair generalizes to every α:** prediction-only leaves composition_error high/noisy
   (2.4–13.7) at all α; H→W drives it to ~0.002 (tiny CI) everywhere. Robust claim = "structural
   H→W installs a real group action"; pure shortcut-cheating is the corner case.

## Phase 2 — agentic H (DONE)
A coding subagent, given only the weak harness + the failed-run trace, independently authored a
structural verifier (`harness/verifier.py`). Detection PASS: cheater.pt flagged is_cheating (5/5
seed votes), honest.pt clean (0/5). Real agent-produced diff: `figures/harness_update.diff` (178 lines).
Detection logic: 3 independent shortcut signatures (solves-broken-symmetry, shortcut-sensitivity,
composition-law violation), scale-relative thresholds, majority vote across seeds.

## Phase 3 — lever-attribution prototype (DONE, 10 seeds × 3 modes = 30 episodes)
Three genuinely-failing (model, harness) states, each needing a different lever. **Measurement and
scoring are separated**: raw true-scores come from REAL reruns (apply lever → retrain/swap →
measure); the W-update cost only re-scores those raw numbers, so the W_COST sweep needs zero extra
training and there is no transition table.

| Mode | starting state | correct lever |
|---|---|---|
| shortcut_leak | cheater + weak harness | H_THEN_W |
| model_prior_gap | honest-but-undertrained + valid harness | W |
| bad_verifier | honest model + buggy harness | H |

Regret = best-achievable true-score (over levers, real reruns) − chosen lever's true-score.
true-score = capability(clean_mse) × honesty(composition_error); W_COST = per-W-retrain penalty
(transparent preference for the cheaper lever when both reach the same quality — the SIA thesis).

### Headline (W_COST = 0.05), `results/phase3_table.md`
| Policy | lever accuracy | mean regret |
|---|---|---|
| W_only | 0.33 (10/30) | 0.421 |
| H_only | 0.33 (10/30) | 0.318 |
| alternating | 0.67 (20/30) | 0.148 |
| **selector (oracle-sandwich)** | **0.97 (29/30)** | **0.014** |

Wilcoxon (selector regret < alternating, paired): p = 0.013.

### W_COST sensitivity (mean regret — selector lowest at EVERY cost, incl. 0.00)
| W_COST | H_only | W_only | alternating | selector |
|---|---|---|---|---|
| 0.00 | 0.355 | 0.408 | 0.168 | **0.019** |
| 0.01 | 0.346 | 0.409 | 0.163 | **0.017** |
| 0.03 | 0.331 | 0.414 | 0.155 | **0.015** |
| 0.05 | 0.318 | 0.421 | 0.148 | **0.014** |
| 0.10 | 0.287 | 0.440 | 0.134 | **0.015** |

**Key defense:** selector wins even at W_COST = 0 (no cost preference) — the result is NOT an
artifact of the cost term. Plot: `results/phase3_plot.png`. Selector is label-free: builds a
known-good reference model (positive control), checks whether the deployed harness accepts it
(sandwich → H if not), else mechanism-probes the model (cheat signature → H_THEN_W; honest-but-weak
→ W). Framed as a **three-mode prototype**, not a generalization benchmark.

## Phase 4 — Gemma garnish (BLOCKED — needs GPU / user decision)
- LoRA weight-update on Gemma = the stretch → needs **GPU** (hackathon blocker).
- Gemma-as-LLM-selector (inference) is doable locally but needs pulling an ollama model — paused
  pending user go-ahead. Rule selector already proves the lever-attribution claim without it.

## Bottom line
Phases 0–3 prove the full claim end-to-end with measured numbers and no synthetic data:
shortcut cheating is real, W-only **preserves** it (Welch t, large effect), H→W **repairs** it, and a
principled selector beats fixed lever policies on real-rerun regret across the whole W_COST sweep.
Demoable on CPU in ~3 min. Paper-style writeup: `documentation/writeup.md`.
