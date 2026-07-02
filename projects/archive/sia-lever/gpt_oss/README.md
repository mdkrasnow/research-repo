# gpt_oss/ — gpt-oss-120b lever-selector lane

Trains and evaluates gpt-oss-120b (± LoRA) as the **learned H/W lever selector**, plus the W-lever
training stack. All data derives from the measured cache (`data/out/action_outcome_cache.jsonl`,
real reruns via `experiments/phase3.py`).

```
lever_io.py        shared prompts + parser + measured-outcome scoring (single source of truth)
client.py          provider-agnostic OpenAI-compatible chat client (env-driven)
check_env.py       GPU/endpoint/package readiness report
smoke_infer.py     one-trace JSON smoke (endpoint or local)
prompts/           lever-selector system + user template
providers/ profiles/  SIA provider/profile examples for gpt-oss (.example)
data/
  build_trace_dataset.py   THE spine: real reruns -> action_outcome_cache.jsonl
  build_sft_dataset.py     trace -> cost-adjusted best action (chat SFT)
  build_dpo_dataset.py     chosen=correct lever, rejected=measured-worse action
  build_grpo_prompts.py    prompts + measured reward table for GRPO
rollout/           base + adapter rollouts; parse_action
train/             LoRA SFT/DPO/GRPO (rank 32, bf16/QLoRA), accelerate/deepspeed configs, provenance
eval/              eval_selector, eval_adapter, compare_policies
serve/             vLLM base + adapter servers
```

Quickstart: see `documentation/gpu_runbook.md`. CPU-only pieces (data build, compare_policies on
rule/fixed/paper-style policies) run without an endpoint; base/LoRA columns need GPU or a served
endpoint.

Guardrails: no synthetic transition tables, no fabricated rewards; every outcome is a real rerun.
H edits scaffold/verifier only; W edits adapter weights only. Adapters log
base_model/dataset_hash/git_commit/gpu_info/train_config/eval_results.
