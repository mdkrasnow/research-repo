# SIA-Lever — Demo + Pitch

## One-sentence thesis
SIA gives a self-improving agent two levers — update the **harness (H)** or the **weights (W)** —
but never solved *which lever to pull*. We build a symmetry testbed where the correct lever has
ground truth, and show (measured, no synthetic data) that **W-only preserves shortcut-cheating
while H→W repairs it**, and that a principled selector beats fixed lever policies.

## The 3-minute story
1. **A model cheats and the weak harness can't tell.** Rotation task with a shortcut channel that
   leaks the answer. Prediction-only training → clean_mse ~0 (looks solved). Hidden: it also solves
   an *impossible* broken-symmetry control and violates the group axioms.
2. **W-only PRESERVES the cheat.** More prediction-reward training leaves shortcut_sensitivity ~1.0
   and composition_error ~4.7 unchanged. Still cheating (13/15 seeds).
3. **H→W repairs it.** Upgrade the verifier (structural tests), then retrain: composition_error
   5.06→0.001, shortcut_sensitivity 0.98→0.002, neg_control 0.12→1.07 (now honestly *fails* the
   broken task). clean_win 15/15 seeds. Welch t vs W-only: all structural metrics p<3e-3, d 1.3–16.
4. **The H update is agent-produced.** A coding subagent, given only the weak harness + the failed
   trace (forbidden from reading the reference verifier), wrote the structural verifier itself
   (`figures/harness_update.diff`, prompt in `phase2-provenance.md`).
5. **Lever choice is the real problem.** Across 3 real failure modes a principled selector gets
   ~10× lower regret than W-only and beats naive alternating — decided from an oracle-sandwich probe,
   no peeking at ground truth, no transition table. Holds across the whole W_COST sweep.

## Money numbers (all measured, real reruns)
Phase 1 — 4-stage episode, **15 seeds** (`results/episode_table.md`, `episode_plot.png`):
| Stage | clean_mse | neg_control_mse | shortcut_sens | composition_err | verdict |
|---|---|---|---|---|---|
| pred-only | 0.003 | 0.12 | 0.98 | 5.06 | shortcut_win 13/15 |
| W-only | 0.003 | 0.15 | 1.01 | 4.66 | shortcut_win 13/15 |
| H→W | 0.006 | **1.07** | **0.002** | **0.001** | clean_win 15/15 |

Welch t (W-only vs H→W): shortcut_sens p=1.6e-16 d=16.3; composition p=5.7e-6 d=2.58; gate 15/15.

Phase 3 — lever attribution prototype, **10 seeds × 3 modes = 30 episodes** (`phase3_table.md`,
`phase3_plot.png`); selector lowest regret at EVERY W_COST incl. 0.00:
| Policy | lever acc | mean regret (W_COST=0.05) | regret @ W_COST=0.00 |
|---|---|---|---|
| W_only | 0.33 | 0.421 | 0.408 |
| H_only | 0.33 | 0.318 | 0.355 |
| alternating | 0.67 | 0.148 | 0.168 |
| **selector** | **0.97** | **0.014** | **0.019** |

Wilcoxon selector<alternating p=0.013. Leak-strength sweep (`leak_sweep_table.md`): shortcut-cheat
is adversarial (only α=1.0); H→W's group-structure repair holds at every leak strength.

## Run it (all CPU)
```bash
cd projects/sia-lever/experiments
python run_seeds.py --seeds 15        # Phase 0/1 (~10 min): W-only preserves cheat, H->W learns group; Welch t
python ../harness/verifier.py         # Phase 2 (~10 s): agent-produced verifier flags cheater, passes honest
python phase3.py --seeds 10           # Phase 3 (~8 min): selector beats fixed policies; W_COST sweep
python leak_sweep.py --seeds 5        # Phase 1b (~10 min): leak-strength robustness
```
Quick demo: `run_seeds.py --seeds 3`, `phase3.py --seeds 3` for ~2 min total.

## What is real vs garnish
- **Real, done, measured:** Phases 0–3. Every number is a forward pass / retrain on real data.
  No synthetic traces. No hardcoded transition table (the trap that would make regret circular).
- **Garnish, not done:** Phase 4 Gemma. LLM-as-selector (inference) + LoRA weight-update on a big
  model. LoRA needs GPU (hackathon); the rule selector already proves the claim.

## Why symmetry is the right testbed
Fake progress is *detectable*: a cheater predicts well but fails the group-composition law and
succeeds on a broken-symmetry negative control where an honest learner must fail. That gives the
ground truth needed to score which lever was actually required — the thing generic SIA benchmarks
lack.

## Honest caveats
- Stage-1/2 neg_control has high seed variance (2/15 seeds the pred-only model partially learned
  rotation). Lead with composition_error + shortcut_sensitivity (clean separation); H→W recovery
  is tight (low CI) and unanimous (15/15).
- W_COST encodes a modest, documented preference for the cheaper/safer lever (the SIA point). It's
  an additive per-W constant applied to raw real-rerun scores, not an outcome lookup. The selector
  wins at every W_COST including 0.00, so the result does not depend on it.
- Selector is 0.97 not 1.0 (one miss / 30) — kept honest, not tuned to a suspicious perfect score.
- Phase 3 is a 3-mode prototype (30 episodes), not a generalization benchmark.
