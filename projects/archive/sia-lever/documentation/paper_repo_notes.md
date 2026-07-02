# SIA paper + official repo notes

## Pinned commit
Official repo: `https://github.com/hexo-ai/sia`
Vendored at: `baselines/vendor/sia` — commit **`99db0e87cbe3a67f6fc251d33b72c88ee1edfac5`**
(see `baselines/vendor/sia_commit.txt`). Paper: arXiv 2605.27276.

## What the paper claims (SIA-W+H)
A self-improving agent improves on a benchmark by updating two levers:
- **H (harness/scaffold):** prompts, tools, parsers, retry logic, search code, verifier/evaluator.
- **W (weights):** LoRA / RL / SFT updates of the task-specific model.
A Feedback-Agent observes the trajectory and chooses H vs W; SIA-W+H > SIA-H on LawBench, TriMul
(CUDA), and scRNA-seq denoising. Future work explicitly points at *learning* the action-selection
(H/W) policy from trajectory/action/outcome triples — the gap this project targets.

## What the PUBLIC repo actually exposes (inspected)
- **Harness/orchestration loop only.** Modules: `orchestrator.py`, `config.py`, `profiles.py`,
  `providers.py`, `backends/` (claude, openhands, pydantic_ai), `tasks/`, `web/`.
- **NO weight-update code.** `grep -rilE "lora|peft|grpo|trl|bitsandbytes"` over `sia/` returns **0
  files**. There is no LoRA/RL/SFT trainer in the public repo. => The public SIA you can run is
  effectively **SIA-H** (harness self-improvement); W and W+H must be implemented ourselves.
- **Bundled tasks with data:** `gpqa`, `lawbench`, `longcot-chess`, `spaceship-titanic`. LawBench
  ships a labeled `train.csv` (usable for our W lane), 913-case test, 191 labels, and an evaluator.
  (TriMul and denoising are **not** ready-to-run bundled tasks at this commit.)

## Custom-task contract (how we plug SIA-Lever in)
- Layout: `tasks/<name>/data/public/{task.md,evaluate.py,...}`, `tasks/<name>/data/private/...`,
  `tasks/<name>/reference/reference_target_agent.py`.
- **Evaluator:** `data/public/evaluate.py` must expose `evaluate(submission_path: Path) -> dict`.
  The orchestrator finds the submission in the generation dir, calls `evaluate`, writes `results.json`,
  and feeds metrics into the next feedback prompt. Standalone `--gen-dir` is also supported.
- **Target agent:** reads public data, queries a model, writes `submission.json` (or `submission.csv`)
  to the generation dir.

## Providers / profiles (exact schema)
- Provider JSON: `{provider_id, name, client_kind: "openai", base_url, api_key_env}`. A **Nebius**
  provider already ships (`api_key_env: NEBIUS_API_KEY`, OpenAI-compatible base_url) — Nebius Token
  Factory serves gpt-oss-120b, so we reuse this client kind.
- Profile JSON: `{profile_id, name, backend: "codegen"|"openhands", model, provider_id}`.
- Our examples: `gpt_oss/providers/gpt_oss_openai_compatible.json.example`,
  `gpt_oss/profiles/gpt_oss_target.json.example` (copy into the SIA config dir, drop `.example`).

## Install (for actually running the loop)
```bash
cd baselines/vendor/sia
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[claude]'   # or '.[openhands]'
sia --help                   # confirm subcommand/flag names for this commit
```

## What we must implement ourselves
1. The **W lever** entirely (LoRA SFT/DPO/GRPO) — done in `gpt_oss/train/` and `paper_benchmarks/lawbench/`.
2. The **learned H/W action-selection policy** and its evaluation — the project's core contribution.
3. A measured **lever-attribution benchmark** (SIA-Lever) with real-rerun outcomes — `sia_task/`.
