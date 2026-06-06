#!/usr/bin/env python3
"""GRPO LoRA on gpt-oss-120b. Reward = the REAL measured cost-adjusted outcome of the sampled
action (looked up in reward_by_action from the cache). No fabricated reward signal.

Data: gpt_oss/data/out/grpo_prompts_train.jsonl (prompt_messages, reward_by_action, token_to_key,
w_retrains, w_cost).

This is the stretch lane (GRPO stack is the most finicky). Run only after SFT/DPO work.
Run: accelerate launch --config_file gpt_oss/train/accelerate_config_multih200.yaml \
       gpt_oss/train/train_lora_grpo.py --base-model openai/gpt-oss-120b --out adapters/.../lever_grpo_<ts>
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "gpt_oss"))
from provenance import write_provenance, gpu_info  # noqa: E402
from lever_io import parse_action  # noqa: E402

DEFAULT_DATA = os.path.join(PROJ, "gpt_oss", "data", "out", "grpo_prompts_train.jsonl")


def load_prompts(path, tok):
    from datasets import Dataset
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            prompt = tok.apply_chat_template(r["prompt_messages"], tokenize=False,
                                             add_generation_prompt=True)
            rows.append({"prompt": prompt, "reward_by_action": json.dumps(r["reward_by_action"]),
                         "token_to_key": json.dumps(r["token_to_key"]),
                         "w_retrains": json.dumps(r["w_retrains"]), "w_cost": r["w_cost"]})
    return Dataset.from_list(rows)


def make_reward_fn():
    def reward_fn(prompts, completions, reward_by_action, token_to_key, w_retrains, w_cost, **kw):
        out = []
        for comp, rba_s, t2k_s, wr_s, wc in zip(completions, reward_by_action, token_to_key,
                                                w_retrains, w_cost):
            rba = json.loads(rba_s); t2k = json.loads(t2k_s); wr = json.loads(wr_s)
            action, _, valid = parse_action(comp if isinstance(comp, str) else comp[-1]["content"])
            if action is None:
                out.append(-1.0); continue          # unparseable -> strong penalty
            key = t2k.get(action)
            if key in rba:
                out.append(float(rba[key]) - float(wc) * float(wr.get(action, 0)))
            else:
                out.append(0.0)                       # PROMOTE/KILL -> documented floor
        return out
    return reward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=os.getenv("GPT_OSS_MODEL_PATH", "openai/gpt-oss-120b"))
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(PROJ, "adapters", "gpt_oss_120b",
                                                  f"lever_grpo_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import GRPOTrainer, GRPOConfig

    cfg = vars(args).copy(); cfg["gpu_info"] = gpu_info()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mk = {"torch_dtype": torch.bfloat16}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        mk["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **mk)

    peft_cfg = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    ds = load_prompts(args.data, tok)
    if args.smoke:
        ds = ds.select(range(min(4, len(ds))))

    grpo_cfg = GRPOConfig(
        output_dir=args.out, num_train_epochs=(0.0 if args.smoke else args.epochs),
        max_steps=(2 if args.smoke else -1), num_generations=args.num_generations,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=args.lr, logging_steps=1, save_strategy="epoch", bf16=True,
        max_completion_length=128,
        report_to=(["wandb"] if os.getenv("WANDB_API_KEY") else []))

    trainer = GRPOTrainer(model=model, args=grpo_cfg, train_dataset=ds,
                          reward_funcs=make_reward_fn(), peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    write_provenance(args.out, args.base_model, args.data, cfg)
    print(f"adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
