# kernel_task — the SIA H-vs-W lever on real GPU kernels (TriMul)

The lever phenomenon (weak verifier passes a fast-but-WRONG kernel; W-only entrenches it; H→W
repairs it) on the SIA paper's actual GPU-kernel benchmark: the AlphaFold-3 **Triangle Multiplicative
Update**, written + optimized as a kernel. CPU-runnable now (stub); gpt-oss + GPU when available.

## Why this exists
The rotation toy + the 4-number lever-attribution task are trivially solvable (see
`documentation/hard_task_design.md`). This task is real: a kernel is "fast because it's wrong"
(skips work / hardcodes the test), which is a genuine, non-toy shortcut. The H lever = strengthen the
correctness verifier; the W lever = the agent rewrites the kernel.

## Run it now (CPU, no model/GPU)
```bash
python kernel_task/check_env.py            # readiness
python kernel_task/tests/test_cpu_stub.py  # phenomenon regression (all PASS)
python kernel_task/run.py                  # 4-policy comparison -> results/kernel/comparison.md
bash scripts/run_kernel_comparison.sh
```
CPU result (stub agent): W-only ends fast-but-WRONG; H-only exposes but can't repair; **H_THEN_W and
the selector reach a correct kernel**. The selector picks H_THEN_W from the implausible-speedup
shortcut signature — no held-out labels.

## Run it on GPU (gpt-oss writes the kernels)
```bash
python kernel_task/run.py --endpoint --device cuda --shape 128 128 32 \
    --model "$GPT_OSS_MODEL" --base-url "$GPT_OSS_BASE_URL"
# or: bash scripts/run_kernel_comparison.sh --endpoint
```
Needs: an H100/H200 with `triton` (see `gpt_oss/requirements-gpu.txt`) + an OpenAI-compatible
endpoint serving gpt-oss-120b.

## Files
- `reference.py`     ground-truth op + input generator (fixed input vs random inputs).
- `seeds/`           candidate kernels: `naive_loops` (slow-correct), `torch_bmm` (fast-correct
                     baseline), `memorize_cheat` (fast-WRONG), `triton_trimul` (real Triton, GPU-only seed).
- `runner.py`        executes a candidate in an ISOLATED subprocess (hard timeout) — untrusted
                     LLM/Triton kernels can hang/segfault; a crash is recorded as "does not run".
- `harness.py`       the H lever: weak (fixed input) vs strong (random inputs) verifier + timing +
                     speedup-vs-baseline. Separates the observable trace from hidden held-out truth.
- `agent.py`         the W actor: `stub` (CPU, deterministic) or `endpoint` (gpt-oss writes a kernel;
                     `extract_code` pulls the python block).
- `lever_loop.py`    the SIA episode + the label-free kernel selector (`select_lever`).
- `run.py`           CLI: 4-policy comparison -> `results/kernel/`.
- `prompts/`         kernel-writer system + user template.

## How the lever maps (vs the rotation toy)
| | rotation toy | kernel task |
|---|---|---|
| shortcut/cheat | reads leaked target | kernel hardcodes/overfits the test input |
| weak verifier | clean-MSE only | correctness on ONE fixed input |
| strong verifier (H) | + neg-control + group axioms | correctness on RANDOM inputs, tight tol |
| W | retrain MLP | agent rewrites the kernel |
| honest test | neg_control_mse | held-out correctness on random inputs |
| selector tell | "too good" on broken control | implausible speedup under the weak verifier |

## GPU-day checklist
1. `python kernel_task/check_env.py` → triton present, cuda True, endpoint creds SET.
2. `python kernel_task/tests/test_cpu_stub.py` → all PASS (loop intact).
3. Smoke: `python kernel_task/run.py --endpoint --device cuda --shape 32 32 8` (small, 1 episode each).
4. Scale shape (128 128 32), inspect `results/kernel/comparison.md`.
5. Headline = does the agent (under the weak verifier) produce a fast-but-wrong kernel, and does
   H_THEN_W / the selector recover a correct one with a real speedup? Report speedup + correctness.

## Honest scope
- The CPU stub agent is deterministic (proves the harness/loop). The real test is whether gpt-oss
  *actually* cheats under the weak verifier — that needs the endpoint.
- This matches the paper's TASK (TriMul Triton), not its RESULT (14× on H100). Report measured
  speedups; do not claim paper reproduction.
- `triton_trimul` seed is correct-but-unoptimized; it is a starting point, not a tuned kernel.
