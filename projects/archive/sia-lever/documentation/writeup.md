# SIA-Lever: a controlled testbed for harness-vs-weight attribution in self-improving agents

*Working note — hackathon project, paper-style writeup. All numbers are measured (real training
reruns); no synthetic traces, no hardcoded outcome tables.*

## 1. Problem
Self-Improving Agents (SIA) expose two levers for improvement: update the **harness (H)** — the
scaffold around a model: prompts, tools, verifier, parsers, search code — or update the **weights
(W)** of the task model. The open question SIA leaves unanswered is *attribution*: when a run
fails, which lever should be pulled? Pulling the wrong lever wastes compute and can entrench
failure (e.g. optimizing weights against a verifier that rewards the wrong thing).

We build a minimal, fully-measured testbed where the correct lever has ground truth, using
**symmetry discovery** as the domain because fake progress is detectable: a model can minimize
prediction error by exploiting a shortcut while failing the group axioms a true symmetry-learner
must satisfy.

## 2. Setup
**Task.** 2D rotation: given a point `x = r[cosθ, sinθ]` and angle `Δ`, predict `y = rotate(x, Δ)`.
The model input carries a **shortcut channel** that leaks the target `y` (fully, or partially as
`α·y + (1−α)·noise`). A prediction-only learner can copy the shortcut and ignore rotation.

**Model.** Tiny `encoder → A(Δ) → decoder`, where `A(Δ)` is a learned 2×2 action matrix. Exposing
the action matrix lets us test group structure directly.

**Metrics (structural battery).**
- `clean_mse` — prediction error (capability).
- `neg_control_mse` — error on a *broken-symmetry* control (target = random point, shortcut leaks
  it). An honest rotation-learner MUST fail this (high = honest); a shortcut-reader succeeds (low).
- `shortcut_sensitivity` — error increase when the shortcut channel is randomized (high = reliant).
- group-axiom errors on `A`: `composition` `‖A(d1)A(d2)−A(d1+d2)‖²`, `identity` `‖A(0)−I‖²`,
  `inverse` `‖A(−d)A(d)−I‖²`. (These replace an earlier ill-posed equivariance proxy.)

This is an **adversarial shortcut trap**, deliberately constructed so a weak (prediction-only)
harness is fooled — not a naturalistic symmetry benchmark.

## 3. Phase 0–1: the core lever phenomenon (15 seeds)
Four stages, each a real training run: (1) prediction-only; (2) **W-only** continue training;
(3) **H** — re-score the same model under the structural verifier; (4) **H→W** — adopt the
structural objective and retrain.

Result (15 seeds, 2000 steps/stage, mean ± 95% CI):

| Stage | clean_mse | neg_control_mse | shortcut_sens | comp_err | id_err | inv_err | verdict |
|---|---|---|---|---|---|---|---|
| pred-only | 0.003±0.004 | 0.116±0.128 | 0.975±0.039 | 5.06±1.55 | 1.71±0.48 | 3.06±1.55 | shortcut_win 13/15 |
| W-only | 0.003±0.004 | 0.145±0.172 | 1.009±0.048 | 4.66±1.41 | 1.67±0.48 | 2.73±1.44 | shortcut_win 13/15 |
| H→W | 0.006±0.004 | 1.067±0.014 | 0.002±0.001 | 0.001±0.000 | 0.32±0.08 | 0.31±0.08 | clean_win 15/15 |

Gate (per-seed S4 neg_control > S2 neg_control): **15/15**. Welch t-test, W-only vs H→W:

| metric | t | p | Cohen's d |
|---|---|---|---|
| shortcut_sensitivity | 44.6 | 1.6e-16 | 16.3 |
| composition_error | 7.06 | 5.7e-6 | 2.58 |
| identity_error | 5.93 | 3.0e-5 | 2.17 |
| inverse_error | 3.61 | 2.8e-3 | 1.32 |

**Claim (defensible):** W-only **preserves** the shortcut failure (structural errors unchanged,
verdict stays shortcut_win); **H→W repairs** it (every structural error collapses to ~0, neg_control
becomes honest). We do not claim W-only makes cheating *worse* — the metrics show persistence, not
significant worsening.

## 4. Robustness to leak strength (Phase 1b) — two findings
Shortcut = `α·target + (1−α)·noise`, swept α ∈ {1.0, 0.75, 0.5, 0.25, 0.0}, 5 seeds. The sweep
splits the result into a corner case and a robust core (`results/leak_sweep_table.md`,
`leak_sweep_plot.png`):

1. **Shortcut-cheating is adversarial, only at α=1.0.** Reading the leak dominates only under a
   perfect leak (shortcut_win 0.80; shortcut_sens 0.997; neg_control 0.17 = dishonest). At α≤0.75
   the noisy leak is no longer the easy path: prediction-only stops using it (shortcut_sens ≈ 0,
   neg_control ≈ 1.0 = honest, shortcut_win 0.00). This **confirms** Phase 1 should be read as an
   adversarial trap, not a generic claim that prediction-only always cheats.
2. **The H→W repair generalizes to every α.** At all leak strengths the prediction-only model
   leaves the learned action structurally broken (composition_error high and seed-noisy: 2.4–13.7),
   while H→W installs a clean group action (composition_error ≈ 0.002 with tiny CI) at every α. The
   robust, leak-independent claim is therefore: *the structural objective (H→W) is what installs a
   real group action; the prediction-only objective does not, whether it cheats via shortcut (α=1)
   or merely fits an incoherent action (α<1).*

## 5. Phase 2: the H update can be agent-produced
Starting from a weak prediction-only verifier that actually *prefers* the cheater
(`cheater.pt` clean_mse 0.000152 < `honest.pt` 0.002478), a coding subagent — given only the weak
harness and the failed-run trace, and forbidden from reading the reference verifier — authored a
structural verifier that flags the cheater (5/5 seed votes) and passes the honest model (0/5).
Full prompt + before/after + detection output: `documentation/phase2-provenance.md`,
diff `figures/harness_update.diff`.

## 6. Phase 3: lever-attribution prototype
<!-- FILL from results/phase3_table.md -->
Three real failure modes, each needing a different lever, every score from real reruns:
`shortcut_leak`→H_THEN_W, `model_prior_gap`→W, `bad_verifier`→H. A label-free selector
(oracle-sandwich: does the deployed harness accept a known-good reference model? + mechanism probe)
is compared against fixed policies. We separate **measurement** (raw true-scores from real reruns)
from **scoring** (a transparent W-update cost), and sweep the cost `W_COST ∈ {0,0.01,0.03,0.05,0.10}`
to show the ranking is not an artifact of the cost choice.

This is explicitly a **three-mode prototype**, not a large benchmark — it motivates lever
attribution and shows it is measurable with real interventions, but does not establish that
selectors generalize across many failure modes.

## 7. Limitations
- Adversarial toy, not naturalistic symmetry learning.
- Phase 3 is a 3-mode prototype (30 episodes at 10 seeds); not a generalization claim.
- `W_COST` is a modeling choice (compute + regression risk); we report the full sensitivity sweep
  and raw (cost-free) scores so the reader can judge.
- The selector uses a positive-control reference model (cheap here; expensive in general).
- Phase 4 (Gemma LLM-selector / LoRA weight-update on a real model) is unfinished — needs GPU.

## 8. The safe claim
> If the harness only rewards prediction, weight updates **preserve** shortcut behavior; after a
> harness update adds structural (group-axiom) checks, a weight update learns the real group action.
> Lever attribution — choosing H vs W — is therefore a real, measurable problem, and a simple
> principled selector beats fixed lever policies on a three-mode prototype.
