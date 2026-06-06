# Reproduction limits (honesty ledger)

State plainly what is and is not reproduced.

## Official paper benchmarks
- **Not reproduced yet.** No LawBench/TriMul/denoising paper numbers have been reproduced at the time
  of this scaffold. Scripts are prepared (`paper_benchmarks/lawbench/`, `baselines/official_sia/`),
  but require GPU + endpoint to run.
- When run, LawBench comparisons are valid only with the **same split (913), label set (191), task
  definition, and a comparable budget**. Any `--limit` run is **reduced LawBench** and excluded from
  paper-headline comparison. Our target model is **gpt-oss-120b**, which differs from the paper's.

## Does the public SIA repo expose W-update code?
- **No.** Inspected at commit `99db0e87…`: `grep -rilE "lora|peft|grpo|trl|bitsandbytes"` over `sia/`
  → **0 files**. The public repo runs the **harness (H) loop only**. Therefore any "W" or "W+H" we
  report is **our own implementation**, labeled paper-STYLE, not an exact reproduction.

## What we implemented ourselves
- The entire **W lever**: LoRA SFT/DPO/GRPO trainers for gpt-oss-120b (`gpt_oss/train/`), plus the
  LawBench W lane (`paper_benchmarks/lawbench/`).
- The **learned H/W action-selection policy** and its evaluation (`gpt_oss/rollout`, `gpt_oss/eval`,
  `methods/`).
- The **measured lever-attribution benchmark** SIA-Lever (`sia_task/`, `gpt_oss/data/`), with all
  outcomes from real reruns (`experiments/phase3.py`).
- A **paper-style** `plateau_then_w` scheduler baseline (not the paper's exact code).

## What remains to reproduce LawBench / TriMul / denoising exactly
- LawBench: run official SIA-H with a matched target; run W on the full split; align the evaluator
  and label set to the paper; report against 13.5 / 50.0 / 70.1 only if matched.
- TriMul: not a ready bundled task at this commit — needs a CUDA build/timing+correctness harness.
- denoising: not a ready bundled task — needs the dataset + a game-resistant reconstruction metric.

## Metric honesty (SIA-Lever)
- Regret is **model-quality based** and cost-adjusted (W_COST=0.05); `plateau_then_w` can tie the
  oracle on regret while mis-attributing — reported as-is. The selector's edge is **attribution
  accuracy + W-efficiency**, not lower regret than the paper-style scheduler.
- The toy underlying the measured outcomes is an **adversarial shortcut trap**, not a naturalistic
  symmetry benchmark (carried over from Phases 0–3).
