#!/usr/bin/env python3
"""LoRA-SFT gpt-oss-120b on LawBench train.csv (text -> criminal-charge label).

This is OUR W-update lane for LawBench (public SIA has no weight-update code). Builds a chat dataset
from the labeled train split and trains a rank-32 LoRA. Reuses the gpt_oss SFT recipe.

Run: accelerate launch --config_file gpt_oss/train/accelerate_config_multih200.yaml \
       paper_benchmarks/lawbench/train_lora_sft.py --base-model openai/gpt-oss-120b --out adapters/.../lawbench_sft_<ts>
"""

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJ, "gpt_oss", "train"))
from provenance import write_provenance, gpu_info  # noqa: E402

LAW = os.path.join(PROJ, "baselines", "vendor", "sia", "sia", "tasks", "lawbench")
TRAIN_CSV = os.path.join(LAW, "data", "training_data", "train.csv")
CLASSES = os.path.join(LAW, "data", "public", "classes.json")

SYS = ("You are a Chinese legal expert. Given the facts (事实) of a criminal case, predict the single "
       "criminal charge (罪名) the court convicted the defendant of. Reply with ONLY the charge label.")


def build_dataset(limit=None):
    from datasets import Dataset
    rows = []
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if limit and i >= limit:
                break
            text = r.get("text") or r.get("事实") or ""
            label = r.get("label") or r.get("罪名") or ""
            if not text or not label:
                continue
            rows.append({"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": text[:6000]},
                {"role": "assistant", "content": label},
            ]})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=os.getenv("GPT_OSS_MODEL_PATH", "openai/gpt-oss-120b"))
    ap.add_argument("--out", default=os.path.join(PROJ, "adapters", "gpt_oss_120b",
                                                  f"lawbench_sft_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    cfg = vars(args).copy(); cfg["gpu_info"] = gpu_info(); cfg["classes"] = CLASSES
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mk = {"torch_dtype": torch.bfloat16, "use_cache": False}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        mk["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **mk)
    model.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    ds = build_dataset(limit=(8 if args.smoke else args.limit))
    sft = SFTConfig(output_dir=args.out, num_train_epochs=(0.0 if args.smoke else args.epochs),
                    max_steps=(2 if args.smoke else -1), per_device_train_batch_size=1,
                    gradient_accumulation_steps=8, learning_rate=args.lr, bf16=True,
                    gradient_checkpointing=True, logging_steps=5, save_strategy="epoch",
                    max_seq_length=4096, packing=False,
                    report_to=(["wandb"] if os.getenv("WANDB_API_KEY") else []))
    trainer = SFTTrainer(model=model, args=sft, train_dataset=ds, peft_config=peft_cfg,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(args.out); tok.save_pretrained(args.out)
    write_provenance(args.out, args.base_model, TRAIN_CSV, cfg)
    print(f"adapter -> {args.out}")


if __name__ == "__main__":
    main()
