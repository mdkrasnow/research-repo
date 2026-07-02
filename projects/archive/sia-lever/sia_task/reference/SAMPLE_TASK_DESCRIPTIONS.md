# SIA-Lever — sample task descriptions for the Meta-Agent

Short framings the SIA Meta-Agent can use when initializing a target agent for this task.

## One-liner
"Choose the correct self-improvement lever (H / W / H_THEN_W / PROMOTE / KILL) for each failed-run
trace, optimizing measured regret — not clean prediction error."

## Medium
"You are the Feedback-Agent of a self-improving system. Each item is a failed run described by a
label-free trace (clean MSE, negative-control MSE, composition error, a shortcut-cheat flag, and an
oracle-sandwich check of whether the harness accepts a known-good model). Decide whether to fix the
harness (H), train the weights (W), do both (H_THEN_W), promote, or kill. A weak harness that rewards
a shortcut will make a weight update worse; a broken harness must be fixed before training. Output
strict JSON per trace and write submission.json."

## Reference agent
`reference/reference_target_agent.py` queries an OpenAI-compatible gpt-oss-120b endpoint with the
shared lever-selector prompt (`gpt_oss/prompts/`), parses strict JSON, and writes submission.json.
Use `--dry-run` to test plumbing without the model.

## Scoring
Hidden subset; metrics = lever_accuracy, mean_regret, max_regret (real-rerun outcomes),
invalid_json_rate, per-mode accuracy. PROMOTE/KILL score zero (every episode is a real failure).
