# TriMul GPU-kernel lane — SIA-W+H (paper-style) vs SIA-Lever

Goal: run the SIA paper's **TriMul GPU-kernel** task (AlphaFold-3 Triangle Multiplicative Update)
for real on a GPU, as faithfully as the public artifacts allow, and compare the paper's H-vs-W
scheduler against our SIA-Lever selector on the same measured episodes.

## What "as faithful as possible" means (and its honest limits)

The public SIA repo (pinned `99db0e87…`) ships the **harness (H) loop only** — no W code, and TriMul
is **not** a bundled runnable task. So an *exact* reproduction is impossible. We reconstruct:

- the **real op + kernels + timing** (faithful): `experiments/trimul_gpu.py`
- the **lever phenomenon** as a measured cache (faithful, real reruns): `experiments/trimul_task.py`
- the **SIA custom-task wiring** (faithful to the contract): `sia_task_trimul/`
- the paper's **H-vs-W Feedback-Agent** as a paper-STYLE scheduler (NOT exact): `plateau_then_w`

Not matched to the paper: exact kernel sizes/inputs/tolerances, absolute latencies, the target model
(we use gpt-oss-120b), and the paper's private W+H code. See `reproduction_limits.md`.

## Components

| File | Role |
|---|---|
| `experiments/trimul_gpu.py` | Real op + candidate kernels (einsum/bmm/**Triton**/loop/memorize/approx), **CUDA-event** timing (perf_counter on CPU), held-out verifier, `build_lat_table`. GPU + CPU. |
| `experiments/trimul_task.py` | Builds the measured lever cache. `--real-latency [--device cuda]` swaps the synthetic latency table for device measurements. Cache schema = unchanged (feeds SFT/eval/compare). |
| `sia_task_trimul/` | SIA custom task: `build_task_data.py`, `data/public/{task.md,evaluate.py}`, `reference/reference_target_agent.py`. Drop into a SIA `tasks/` dir to run the real H-loop. |
| `methods/kernel_lever_rule.py` | SIA-Lever deterministic rule on the numeric kernel trace (label-free analog of oracle_sandwich). |
| `gpt_oss/eval/compare_sia_wh_vs_lever.py` | Head-to-head + pos/neg controls -> `results/trimul_sia_wh_vs_lever.{csv,md}`, `plots/...png`. |
| `scripts/run_trimul_gpu_comparison.sh` | One command, CPU-fallback, `--real-latency` on GPU. |

## Run

```bash
# CPU (anywhere) — synthetic latency, full comparison
bash scripts/run_trimul_gpu_comparison.sh

# GPU box (H200/A100) — real CUDA-event latency + Triton kernel
bash scripts/run_trimul_gpu_comparison.sh --real-latency --reps 10

# add the learned SIA-Lever (gpt-oss ± LoRA) column once rollouts exist
bash scripts/run_trimul_gpu_comparison.sh --lever-rollouts 'results/gpt_oss/sft_rollouts_*.jsonl'
```

To run the actual SIA H-loop on TriMul: copy `sia_task_trimul/` into the vendored SIA `tasks/trimul/`
dir (keeping the data/public + reference layout), point a profile at gpt-oss-120b, and invoke `sia`.

## Result (CPU, single run — illustrative, not paper numbers)

| policy | lever_acc | mean_regret | w_calls |
|---|---|---|---|
| oracle_best (POS) | 1.00 | 0.000 | 72 |
| W_only (NEG) | 0.00 | 0.763 | 96 |
| H_only (NEG) | 0.25 | 0.525 | 0 |
| **sia_wh_plateau** (paper-style) | 1.00 | 0.000 | 72 |
| **sia_lever_rule** (ours) | 0.75 | 0.175 | 48 |

**Reading.** Paper-style SIA-W+H ties the oracle — but *because it re-measures H's outcome* (a real
H rollout it gets to pay for). SIA-Lever decides from the **trace alone** (no re-measurement, 48 vs
72 W-calls) and the label-free rule leaves a 0.75->1.00 accuracy gap. That gap is the target of the
**learned LoRA selector** (GPU/endpoint blocked). The honest claim is "names the minimal correct
lever more cheaply," not "lower regret than a competent re-measuring scheduler."

## Next (GPU)

1. `--real-latency` cache on H200; confirm the lever phenomenon holds with measured latencies.
2. gpt-oss-120b base + LoRA rollouts on `sia_task_trimul` traces -> add `sia_lever_lora` column.
3. Headroom: if `trace_difficulty_probe` shows a tiny rule already = 1.00, the learned win is
   meaningless — report honestly (this mirrors the rotation-toy lane's documented limited headroom).
