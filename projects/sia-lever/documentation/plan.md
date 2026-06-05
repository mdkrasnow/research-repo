# SIA-Lever — Phased Plan

## Core claim (prove ONCE, measured)
**W-only updates PRESERVE shortcut-cheating. H-first (fix verifier) then W repairs it.**
This is the real SIA lever phenomenon. Real reruns, real deltas — not asserted, not synthetic.
(Defensible wording: W-only "preserves/keeps" the shortcut — the numbers show persistence, not
clear worsening. Avoid "reinforced/made worse" unless a metric clearly increases.)

## SIA framing
Self-improving agent has two levers:
- **H = harness update** — change prompts, tools, verifier, parser, search/scaffold code. Weights fixed.
- **W = weight update** — change model params (LoRA/finetune). Scaffold fixed.

Open problem (the research angle): **when a run fails, which lever?** Symmetry toy gives ground truth
because fake progress (shortcut) is detectable: model predicts well but fails group-structure tests.

## Anti-goals (what killed the prior plan)
- NO synthetic hand-authored traces — nothing ever really fails → no real failure modes.
- NO hardcoded `TRANSITION_EFFECTS` table — that table IS the oracle → regret circular by construction.
- NO Gemma LoRA load-bearing — solo, 1 day, 12B LoRA = high-risk time sink. Garnish only.
- NO 200-episode benchmark before the single real loop is measured.

---

## Phase 0 — Toy + measurement (HOP) — must work first
Smallest real loop. CPU, seconds per train, no big model.

Files: `data.py`, `model.py`, `verifier.py`, `train.py`.

Task: 2D point `x = r[cosθ, sinθ]`, rotation `Δ`, target `y = r[cos(θ+Δ), sin(θ+Δ)]`.
Input to model = `[x_x, x_y, sinΔ, cosΔ, shortcut_x, shortcut_y]` where shortcut channel
leaks the target `[cos(θ+Δ), sin(θ+Δ)]`. Cheater reads shortcut, ignores rotation structure.

Negative control: target `y_fake = r[cosψ, sinψ]`, ψ random unrelated to θ+Δ, shortcut leaks ψ.
True learner FAILS neg control. Shortcut learner SUCCEEDS → that's the tell.

Metrics (all measured):
- `clean_mse` — prediction on normal rotated targets
- `neg_control_mse` — on broken-symmetry control. HIGH = good (no real symmetry to exploit honestly)
- `shortcut_sensitivity` — MSE jump when shortcut channel randomized
- `composition_error` — `||A_Δ1 A_Δ2 − A_(Δ1+Δ2)||` (group composition axiom)
- `identity_error` — `||A_0 − I||` ; `inverse_error` — `||A_-Δ A_Δ − I||` (group axioms)
  (these replaced an earlier ill-posed equivariance proxy that rose after H→W)

**Exit gate:** prediction-only model → low `clean_mse` AND low `neg_control_mse`.
That confirms shortcut win is REAL. If `neg_control_mse` already high → toy not trapping, fix shortcut.

## Phase 1 — Four-stage episode (HOP, measured) — THE WIN
Run, record real deltas:

| Stage | Action | Expect |
|---|---|---|
| 1 | train prediction-only (verifier v0) | clean low, neg-ctrl low = cheat |
| 2 | W-only continue (more prediction reward) | cheat same/worse |
| 3 | H update: verifier v1 adds neg-ctrl + composition | shortcut now detected |
| 4 | H→W: retrain vs structural objective | clean low, neg-ctrl HIGH, comp down |

Money line, MEASURED: **"W-only preserved the shortcut failure; H-then-W repaired it."**

**Exit gate:** stage-4 `neg_control_mse` > stage-2 `neg_control_mse`, real reruns.
If only this lands, project is a win. Everything after = bonus.

## Phase 2 — Agentic H (WALK)
Make the H update agent-produced, not hand-written. Claude Code reads stage-2 failed trace,
patches verifier: adds `composition_test`, `negative_control`, `shortcut_detector`.
Save real diff `figures/harness_update.diff`.

**Exit gate:** visible agent diff + detection works on patched harness.

## Phase 3 — Lever selector (RUN) — only if 0+1+2 done with time
2-3 REAL failure modes, each a run that really fails:
- `shortcut_leak` → H_THEN_W
- `model_prior_gap` (clean task, oracle passes, model fails) → W
- `parser_bug` / `bad_verifier` → H

Selector reads real trace → picks action. Regret = apply action + RE-RUN (not lookup table).

**Exit gate:** rule-selector beats H-only and W-only on ≥3 real episodes, measured regret.

## Phase 4 — Gemma garnish (FLY) — last, droppable
Gemma 3 12B (verify version — "Gemma 4" likely confabulation) as trace explainer / lever selector /
LoRA target. NEVER load-bearing. LoRA breaks → fallback tiny classifier, stated honestly.

---

## Build order
**0→1 = the win. 2 = polish. 3 = research story. 4 = garnish.**
Phenomenon first. Benchmark/selector/Gemma only as honestly-earned additions after the hop lands.
