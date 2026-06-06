# Positioning: the next layer is lever attribution

## Title
**SIA-Lever-120B: Learning When to Update Harness vs Weights in Self-Improving Agents.**

## The argument
1. SIA proved **H+W beats H-only**: both harness updates and weight updates help.
2. The unsolved next problem is **attribution** — when a run fails, *which* lever?
   The paper's Feedback-Agent makes this choice with a **frozen LLM prior**, and its future work
   explicitly proposes **learning the action-selection policy** from trajectory/action/outcome triples.
3. **SIA-Lever supplies exactly those triples with REAL measured outcomes.** Each episode is a genuine
   failure; for every lever we actually retrain/re-evaluate and measure the outcome (no transition
   table). So "the best lever" is a measured fact, not a guess.
4. **gpt-oss-120b + LoRA** makes the result paper-aligned: the task-specific model's *weights* are
   updated to choose the lever better — the W lever applied to the meta-decision itself.

## The Goodhart hook (why the benchmark matters)
Both levers optimize the same verifier, so they can **couple-Goodhart**: a weight update against a
weak/broken harness trains the model deeper into a shortcut. SIA-Lever makes this concrete:
- `bad_verifier`: the harness rejects a known-good model — W here trains on bad feedback → must do **H**.
- `shortcut_leak`: a weak harness passed a cheater — must fix the verifier first → **H_THEN_W**.
- `model_prior_gap`: harness valid, model weak → **W**.
A policy that pulls W indiscriminately (W_only) is the worst; our lever-aware policy verifies harness
validity (oracle sandwich) *before* touching weights.

## The defensible claim (do not overclaim)
> On SIA-Lever — a lever-attribution benchmark with real-rerun outcomes designed to expose
> coupled Goodhart — naive fixed policies (H-only/W-only/alternating) incur high regret; a
> paper-style plateau-then-W scheduler reaches low regret but mis-attributes; and a learned
> lever-aware policy (gpt-oss-120b, improved by LoRA) chooses H vs W vs H→W from the trace with
> high accuracy and low regret under the same measured evaluator.

What we do NOT claim:
- We do **not** claim to beat SIA on its paper benchmarks unless we run them on the same split/budget.
- We do **not** claim lower regret than the paper-style scheduler on this benchmark; on the model-
  quality regret metric they can tie. Our edge is **one-shot attribution accuracy** + not over-pulling
  W + explicit harness-validity checking, and that the **model's weights** can be trained to do it.

## Honest nuance from the measured CPU comparison
`plateau_then_w` reaches ~0 regret here (re-measuring after H lets it recover quality) but only ~0.67
lever-accuracy; the oracle-sandwich rule gets 1.00 accuracy / 0 regret. So the headline differentiator
is **attribution accuracy and W-efficiency**, and the gpt-oss(+LoRA) result shows a *model* learning
that attribution — which is the paper's stated future-work direction.
