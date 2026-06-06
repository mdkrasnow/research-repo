# GPT-OSS-120B / H200 pivot

## Why pivot from Gemma to gpt-oss-120b
- The SIA paper's weight-update experiments use **gpt-oss-120b** with **LoRA rank 32**. Matching the
  model + LoRA setup makes our W-lever result paper-aligned and directly comparable in spirit.
- gpt-oss-120b is **open weights** (trainable LoRA) and is served on OpenAI-compatible endpoints
  (e.g. Nebius Token Factory) for cheap base-model rollouts — so we can do both base eval (endpoint)
  and weight updates (H200) without a bespoke stack.
- H200s give the memory headroom for bf16 LoRA on 120B (ZeRO-3 across GPUs; QLoRA fallback on one).

## How this aligns with the paper
| Paper element | Our implementation |
|---|---|
| H lever (scaffold) | our verifier/harness edits (Phase 2 agentic-H; `sia_task` harness) |
| W lever (LoRA r32) | `gpt_oss/train/train_lora_{sft,dpo,grpo}.py` on gpt-oss-120b |
| Feedback-Agent H/W choice (frozen prior) | base gpt-oss-120b selector (`rollout_base`) |
| future work: learn the H/W policy | LoRA-trained gpt-oss selector (`rollout_adapter`) |
| benchmarks (LawBench…) | `paper_benchmarks/lawbench/` + official SIA-H loop |

## What is paper-faithful vs our extension
- **Paper-faithful:** model (gpt-oss-120b), LoRA rank 32, bf16, the H-vs-W framing, LawBench as a
  bundled task, evaluation-after-each-generation loop.
- **Our extension:** (1) a measured lever-attribution benchmark with real-rerun outcomes; (2) the
  *learned* H/W selection policy and its evaluation; (3) the W-lever code itself (public SIA has none).

## What H and W are here
- **H** = edit the harness/verifier (e.g. add the structural/negative-control checks that turn a weak
  prediction-only evaluator into one that detects shortcuts). No weights change.
- **W** = LoRA-update the task model. For the lever task, W trains gpt-oss-120b to map a failure trace
  → the correct lever. For LawBench, W trains gpt-oss-120b on (facts → charge).

## What the LoRA is trained on
- **SFT:** trace → cost-adjusted best lever (JSON), from the measured cache.
- **DPO:** chosen = correct lever, rejected = a measured-worse action (regret gap ≥ tol).
- **GRPO:** reward = the REAL measured cost-adjusted outcome of the sampled action.
All targets come from `gpt_oss/data/out/action_outcome_cache.jsonl` (real reruns via Phase 3). No
fabricated rewards, no transition table.

## Why the selector itself is the next layer
SIA already updates H and W. The open question is the *policy that decides between them*. We make that
policy a trainable artifact (gpt-oss-120b weights) and measure whether training improves it — turning
the paper's "frozen prior chooses the lever" into "the lever-choice is learned and evaluated".
