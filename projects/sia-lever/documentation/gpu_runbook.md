# GPU runbook — SIA-Lever-120B on H200

Exact commands. Everything before §5 runs on CPU and needs no endpoint.

## 0. Env
```bash
cd projects/sia-lever
python3 -m pip install -r gpt_oss/requirements-gpu.txt   # on the GPU box
export GPT_OSS_BASE_URL="https://api.tokenfactory.nebius.com/v1/"   # or your vLLM URL
export NEBIUS_API_KEY=...          # (or GPT_OSS_API_KEY / OPENAI_API_KEY)
export GPT_OSS_MODEL="openai/gpt-oss-120b"   # confirm exact served id via check_env
# for local training/serving instead of a hosted endpoint:
export GPT_OSS_MODEL_PATH=/path/to/gpt-oss-120b   # HF dir
```

## 1. Environment check
```bash
python3 gpt_oss/check_env.py
```

## 2. Smoke inference (one lever trace -> JSON)
```bash
python3 gpt_oss/smoke_infer.py --model "$GPT_OSS_MODEL"        # endpoint
# or local: python3 gpt_oss/smoke_infer.py --local --model "$GPT_OSS_MODEL_PATH"
```

## 3. Build datasets (CPU; from measured cache)
```bash
python3 gpt_oss/data/build_trace_dataset.py --seeds 10 --steps 800   # rebuild cache (real reruns)
python3 gpt_oss/data/build_sft_dataset.py
python3 gpt_oss/data/build_dpo_dataset.py
python3 gpt_oss/data/build_grpo_prompts.py
python3 sia_task/build_task_data.py
```

## 4. Base gpt-oss selector eval
```bash
python3 gpt_oss/rollout/rollout_base.py --model "$GPT_OSS_MODEL" --tag base
python3 gpt_oss/eval/eval_selector.py --rollouts 'results/gpt_oss/base_rollouts_*.jsonl' --tag base
```

## 5. SFT LoRA (rank 32, bf16)
```bash
# multi-H200 (set num_processes in the config to your GPU count):
accelerate launch --config_file gpt_oss/train/accelerate_config_multih200.yaml \
  gpt_oss/train/train_lora_sft.py --base-model "$GPT_OSS_MODEL_PATH" --out adapters/gpt_oss_120b/lever_sft_$(date -u +%Y%m%dT%H%M%SZ)
# single H200 (auto QLoRA fallback on OOM):
bash scripts/run_gpt_oss_lora_sft.sh
# fast wiring check:
EXTRA=--smoke bash scripts/run_gpt_oss_lora_sft.sh
```

## 6. DPO LoRA (optional)
```bash
bash scripts/run_gpt_oss_lora_dpo.sh
```

## 7. Serve adapter + roll out + eval
```bash
bash gpt_oss/serve/serve_vllm_adapter.sh adapters/gpt_oss_120b/lever_sft_<ts> lever_lora   # one shell
export GPT_OSS_BASE_URL=http://localhost:8001/v1 GPT_OSS_API_KEY=dummy
python3 gpt_oss/rollout/rollout_adapter.py --endpoint --model lever_lora --tag sft            # another shell
python3 gpt_oss/eval/eval_adapter.py --adapter-rollouts 'results/gpt_oss/sft_rollouts_*.jsonl' \
                                     --base-rollouts 'results/gpt_oss/base_rollouts_*.jsonl' --tag sft
```
(Or `--local` in rollout_adapter to load base+adapter via PEFT without serving.)

## 8. Final comparison (one command)
```bash
bash scripts/run_gpu_comparison.sh                 # full: env->cpu->data->base->train->adapter->compare
bash scripts/run_gpu_comparison.sh --dry-run       # print the plan
bash scripts/run_gpu_comparison.sh --skip-train --adapter adapters/gpt_oss_120b/lever_sft_<ts>
bash scripts/run_gpu_comparison.sh --limit 9 --model "$GPT_OSS_MODEL"
```
Outputs: `results/final_comparison.{csv,md}`, `plots/final_comparison.png`.

## 9. LawBench stretch
```bash
bash baselines/official_sia/run_lawbench_sia_h.sh
python3 paper_benchmarks/lawbench/train_lora_sft.py --base-model "$GPT_OSS_MODEL_PATH" --out adapters/gpt_oss_120b/lawbench_sft_<ts>
python3 paper_benchmarks/lawbench/eval_lora.py --adapter adapters/gpt_oss_120b/lawbench_sft_<ts>
```

## Resolved-version log (fill in what actually worked)
- torch: …  transformers: …  trl: …  peft: …  vllm: …  bitsandbytes: …  CUDA/driver: …
- GPU launch used: single / multi (ZeRO-3) / QLoRA
- gotchas: …
