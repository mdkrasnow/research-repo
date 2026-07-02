# Kernel (TriMul GPU) LoRA Selector — Provenance

Goal: produce a REAL trained LoRA selector on the GPU-kernel (TriMul) task so the rung-3 result is
a trained model, not the hand-written rule. Token Factory managed fine-tune (API-only, no GPU VM).

## Status: DATA READY + VALIDATED — LAUNCH BLOCKED (no Token Factory API key)

Everything up to the managed fine-tune launch is complete and verified. The launch itself is blocked
because no Nebius / Token Factory API key is present in the environment.

## Data source

- **Cache**: `gpt_oss/data/out/kernel_cache_gpu.jsonl` — REAL CUDA latency, GPU-kernel (TriMul) task.
- 320 episodes, 10 seeds (0–9), 8 modes (`{v5,v50}|{k_memorize,k_loop,k_einsum,k_approx}`).
- Schema identical to hard_cache (trace_text, reward_by_action, correct_action, seed, episode_id, mode).
- Gold label distribution (cost-adjusted best lever), all 320 ep: **H_THEN_W 240 / H 80** (2 classes only).
  Per seed: 24 H_THEN_W + 8 H.

## Split + balancing (builder: `gpt_oss/data/build_kernel_sft.py`)

Mirrors `build_balanced_v2.py`:
- **Eval fold**: held-out seed **9** (full, unbalanced — true deploy distribution): 32 ep = 24 H_THEN_W + 8 H.
- **Train candidates**: seeds 0–8 (288 ep). Class-balanced by **downsampling the majority to the
  minority count**, deterministic **sort-by-episode-id, NO RNG** (same fix that cured the rung-2
  majority-collapse, see `documentation/ladder_findings.md`).
  - Raw candidate counts: H_THEN_W 216 / H 72.
  - Kept 72/class → **train = 144 rows (H_THEN_W 72 / H 72)**.
- **Leakage**: train excludes the eval-seed fold; zero episode_id overlap asserted in the builder
  AND independently by `validate_dataset.py`.

### Validation (`gpt_oss/data/validate_dataset.py`) — PASS
```
train n=144 labels={'H_THEN_W': 72, 'H': 72}  | eval n=32
VALIDATION OK: no leakage, all present labels >= min-frac, non-empty.
```

Upload prep verified via `launch_tf_finetune.py --dry-run`: 144 train / 32 eval cleaned to
OpenAI-format `{"messages": [...]}`-only lines.

## Fine-tune recipe (winning rung-2 recipe)

- Base model: `unsloth/gpt-oss-120b-BF16` (`openai/gpt-oss-120b`)
- learning_rate **3e-5**, **lora_r 32** (lora_alpha 64), **n_epochs 20**, **packing true**
- tag: `kernel_sft_3e5`

## Blocker

`launch_tf_finetune.py` reads the key from `NEBIUS_API_KEY` (or `GPT_OSS_API_KEY` /
`OPENAI_API_KEY`). **None are set** in this environment (only `BRAVE_API_KEY`,
`SUPERMEMORY_CC_API_KEY`). Also needs `GPT_OSS_BASE_URL` (defaults to the Token Factory URL) and
`GPT_OSS_MODEL` (defaults correctly).

### To launch when the key is available
```
python3 gpt_oss/launch_tf_finetune.py \
  --tag kernel_sft_3e5 --epochs 20 --lora-r 32 --learning-rate 3e-5 --packing true \
  --src-train gpt_oss/data/out/kernel_v1/trace_action_train.jsonl \
  --src-eval  gpt_oss/data/out/kernel_v1/trace_action_eval.jsonl \
  --job-out   gpt_oss/data/out/kernel_v1/job.json
# then: python3 gpt_oss/poll_tf_finetune.py --job-id <ftjob-...>
# on success: python3 gpt_oss/download_ft_result.py --job-file gpt_oss/data/out/kernel_v1/job.json --tag kernel_sft_3e5
```

## EVAL is PENDING

Per instruction, this work does NOT serve or evaluate the adapter — the GPU serving VM is down.
Once trained, the adapter is downloaded to `adapters/gpt_oss_120b/` and rung-3 eval (selector
accuracy on the held-out seed-9 fold) remains **PENDING VM serving**. The adapter alone does not
establish a rung-3 result; it must be served + evaluated before any accuracy claim.

## TRAINED (2026-06-06)
- job: `ftjob-85c461bc5e0643f7a00192fc78eb048a` status=**succeeded**, trained_tokens=2,438,440.
- recipe: unsloth/gpt-oss-120b-BF16 base, LoRA r32/alpha64 (attention q/k/v/o), lr3e-5, 20ep, packing=true.
- data: 144 balanced train (72 H / 72 H_THEN_W), 32 held-out eval (seed 9), zero episode_id leakage.
- adapter: `adapters/gpt_oss_120b/kernel_sft_3e5_ftjob-85c461bc5e0643f7a00192fc78eb048a/adapter_model.safetensors` (47.8 MB).
- EVAL: **PENDING** vLLM/VM serving (down). No rung-3 trained-selector accuracy claim until served + evaluated on the seed-9 fold and run through collapse_report.py (2-class set → watch for constant-H_THEN_W collapse).

## EVALUATED (2026-06-07) — trained LoRA NO_WIN
Served base MXFP4 + kernel_lora on the H200 (vLLM, tmux), tunneled, evaluated on the held-out
seed-9 fold (32 ep, gold H_THEN_W 24 / H 8). Apples-to-apples on the SAME fold:
- base gpt-oss-120b: acc 0.250, regret 0.748 (dist W19/H_THEN_W7/H6)
- trained kernel-LoRA: acc 0.500, regret 0.472, dist H24/H_THEN_W8 -> **NO_WIN** (collapse_report):
  beats_base T, regret_not_worse T, not_collapsed T, **beats_constants F** (best constant
  always-H_THEN_W = 0.750 on this fold). LoRA gets approx+einsum (correct-kernel -> H) right but
  FAILS loop+memorize (broken/cheat kernel -> H_THEN_W), so it under-predicts the lever that matters.
- hand-rule on the same fold: acc 0.750, regret 0.236 -> beats the constant, crushes regret.

**Verdict:** the hand-written label-free rule REMAINS the best kernel selector. SFT on 144 balanced
examples improved over base but did not beat the rule or the majority constant. Honest negative for
the trained-LoRA kernel transfer at this data scale. Likely fix = more kernel episodes (144 is tiny
and 2-class) so SFT can learn the broken-kernel -> H_THEN_W mapping; mirrors the rung-2 data-scale
lesson. Not a mechanism failure (base+rule bracket it correctly), a data-scale failure.
