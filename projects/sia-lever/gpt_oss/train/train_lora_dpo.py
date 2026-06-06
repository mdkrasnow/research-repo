#!/usr/bin/env python3
"""DPO LoRA on gpt-oss-120b: prefer the correct lever over a measured-worse action.

Data: gpt_oss/data/out/dpo_pairs_train.jsonl (prompt_messages, chosen, rejected; chosen reward >=
rejected reward, validated at build time). LoRA rank 32, bf16, adapter-only save, provenance.

Run: accelerate launch --config_file gpt_oss/train/accelerate_config_multih200.yaml \
       gpt_oss/train/train_lora_dpo.py --base-model openai/gpt-oss-120b --out adapters/.../lever_dpo_<ts>
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from provenance import write_provenance, gpu_info  # noqa: E402

DEFAULT_DATA = os.path.join(PROJ, "gpt_oss", "data", "out", "dpo_pairs_train.jsonl")


def load_pairs(path, tok):
    from datasets import Dataset
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            prompt = tok.apply_chat_template(r["prompt_messages"], tokenize=False,
                                             add_generation_prompt=True)
            rows.append({"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=os.getenv("GPT_OSS_MODEL_PATH", "openai/gpt-oss-120b"))
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(PROJ, "adapters", "gpt_oss_120b",
                                                  f"lever_dpo_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import DPOTrainer, DPOConfig

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
    model.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    ds = load_pairs(args.data, tok)
    if args.smoke:
        ds = ds.select(range(min(4, len(ds))))

    dpo_cfg = DPOConfig(
        output_dir=args.out, num_train_epochs=(0.0 if args.smoke else args.epochs),
        max_steps=(2 if args.smoke else -1), beta=args.beta,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=1, save_strategy="epoch", bf16=True, gradient_checkpointing=True,
        report_to=(["wandb"] if os.getenv("WANDB_API_KEY") else []))

    trainer = DPOTrainer(model=model, args=dpo_cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    write_provenance(args.out, args.base_model, args.data, cfg)
    try:
        from plot_trainer_log import plot_from_dir
        plot_from_dir(args.out)
    except Exception as e:
        print(f"[warn] training-curve plot failed: {e}")
    print(f"adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
