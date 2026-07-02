# Phase 2 — agentic-H provenance (reproducibility record)

Purpose: defend that the harness (H) update was genuinely produced by a coding agent from the
failed-run trace, NOT hand-coded. Everything needed to reproduce/audit it is listed here.

## Artifacts
- **Before** (weak harness, prediction-only): `harness/verifier_weak_v0_backup.py`
- **After** (agent-authored structural harness): `harness/verifier.py`
- **Diff**: `figures/harness_update.diff` (178 lines)
- **Inputs the agent could read**: `harness/trace_stage2.md` (the failed-run trace + task),
  `experiments/data.py`, `experiments/model.py`. It was told NOT to read `experiments/verifier.py`
  (which already contained a structural verifier) so the patch could not be copied.
- **Checkpoints under test**: `harness/cheater.pt` (W-only, shortcut-reliant),
  `harness/honest.pt` (H→W, real group).

## Exact subagent prompt
The agent was a general-purpose coding subagent (separate context) given this task verbatim:

> You are the harness-update (H) step of a self-improving agent (SIA). A model passed a weak
> verifier but is secretly cheating. Your job: diagnose from the trace and PATCH the verifier to
> detect the cheat.
>
> Working dir: projects/sia-lever
> Read first: harness/trace_stage2.md, harness/verifier.py (the WEAK harness to extend),
> experiments/data.py (make_batch modes clean/neg_control/shortcut_rand), experiments/model.py
> (SymmetryLearner; .action_matrix(delta) exposes the learned group action).
> Then edit ONLY harness/verifier.py to add structural detection. Requirements:
> 1. add verify_structural(model, seed=0) returning clean_mse, neg_control_mse,
>    shortcut_sensitivity, composition_error, and a boolean is_cheating.
> 2. Detection must be principled (not hardcoded to filenames): cheating if predicts clean well AND
>    (solves the broken-symmetry neg control OR action matrices violate composition OR high
>    sensitivity to randomizing the shortcut channel).
> 3. __main__ prints verify_structural for cheater.pt and honest.pt showing cheater is_cheating=True,
>    honest is_cheating=False.
> Do NOT modify experiments/. Verify by running python harness/verifier.py; iterate until robust.

Constraints that make the result defensible:
- Agent could NOT see `experiments/verifier.py` (the reference structural verifier) → no copying.
- Detection rule required to be filename-independent (mechanism-based).
- Agent had to make `python harness/verifier.py` actually pass before finishing.

## What the agent produced (its own design choices)
- Three independent shortcut signatures, flagged if the model predicts clean well AND trips ANY:
  (1) solves the broken-symmetry negative control, (2) error explodes when the shortcut channel is
  randomized (>10× clean error), (3) learned action violates the group composition law
  A(d1)@A(d2)=A(d1+d2) by >10% normalized residual.
- Thresholds expressed RELATIVE to the clean target scale (E‖y‖²≈1), not magic constants.
- Verdict taken as a majority vote across 5 seeds for robustness.

## Detection output (reproduce: `python harness/verifier.py`)
```
=== weak verify() (prediction-only, the harness being patched) ===
cheater.pt {'clean_mse': 0.000152}      <- weak harness PREFERS the cheater
honest.pt  {'clean_mse': 0.002478}

=== verify_structural() (patched: mechanism probes) ===
cheater.pt  ... is_cheating=True   (cheating votes across seeds 0-4: 5/5)
honest.pt   ... is_cheating=False  (cheating votes across seeds 0-4: 0/5)
VERDICT: cheater flagged=True  honest flagged=False  -> detection PASS
```

## Honest scope
This is one agent-produced harness patch, reproducible from the trace + the prompt above. It is not
a fully closed autonomous SIA loop (the orchestration around it is scripted). For the claim "an H
update can be produced by an agent from a failed trace," this is sufficient and auditable.
