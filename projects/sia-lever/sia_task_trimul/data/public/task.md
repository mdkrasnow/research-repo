# SIA-Lever-TriMul — lever attribution on the GPU-kernel task

This is a **SIA custom task** (SIA evaluation contract). It instantiates the SIA paper's
**TriMul GPU-kernel** benchmark (AlphaFold-3 Triangle Multiplicative Update) as a
**lever-attribution** problem: given a *failed-run trace* from a self-improving agent that was
optimizing a CUDA/Triton kernel, decide **which lever to pull**.

## The underlying task (real op)

The agent optimizes the Triangle Multiplicative Update "outgoing" core:

    out[i, j, c] = sum_k  a[i, k, c] * b[j, k, c]            # einsum 'ikc,jkc->ijc'

while preserving correctness. Real candidate kernels (einsum/bmm/Triton/naive-loop) plus two
failure modes — a **memorize cheat** (O(1), hardcodes one fixed input's answer; fast but wrong) and
a **partially-correct approx** kernel — are timed for real (CUDA events on GPU, perf_counter on CPU)
and verified on held-out random inputs. See `experiments/trimul_gpu.py` and
`experiments/trimul_task.py` (`--real-latency` on a GPU box).

## What the agent decides (the lever)

Each episode is a deployed system = (verifier_state, emitted_kernel) that FAILED. Choose one:

- `H`        — fix the verifier (strengthen weak/broken checks), keep the emitted kernel.
- `W`        — re-select the fastest kernel passing the CURRENT verifier (keep the verifier).
- `H_THEN_W` — fix the verifier, THEN re-select the fastest correct kernel.
- `PROMOTE`  — declare solved (no intervention).
- `KILL`     — abandon.

Outcomes are **measured by real reruns** (`data/private/measured_outcomes.jsonl`), not a transition
table. A W costs one retrain (`W_COST=0.05`); the correct lever is the **cost-adjusted minimal**
one.

## Observable trace (what you get)

Numeric, no giveaway booleans:
- `deployed_weak_pass_rate` — pass-rate on loose/weak checks
- `deployed_heldout_pass_rate` — pass-rate on tight held-out random inputs
- `weak_minus_heldout_gap` — large => a cheat passed a weak verifier
- `deployed_latency_us` — emitted kernel latency
- `known_good_kernel_accept_rate` — does the deployed verifier accept a known-good einsum kernel?
  (low => the verifier itself is broken/over-strict)

## Submission

Write `submission.json` (or `results/submission.json`) into the generation dir:

```json
{"predictions": [{"episode_id": "...", "action": "H_THEN_W", "reason": "..."}]}
```

`data/public/evaluate.py` exposes `evaluate(submission_path) -> dict` (lever_accuracy, mean_regret,
max_regret, w_calls implied by action_distribution, invalid_json_rate). The reference target agent is
`reference/reference_target_agent.py` (queries gpt-oss-120b; `--dry-run` for plumbing).

## Build the data

```bash
python experiments/trimul_task.py --reps 10 [--real-latency --device cuda]   # -> kernel_cache.jsonl
python sia_task_trimul/build_task_data.py                                     # -> public/private splits
```
