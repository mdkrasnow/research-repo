#!/usr/bin/env python3
"""SFT LoRA on gpt-oss-120b for the lever-selection task (trace -> best-action JSON).

LoRA rank 32 (SIA paper), bf16, gradient checkpointing, adapter-only save, full provenance.
GPU strategy auto-detected:
  - >=2 GPUs : bf16 LoRA (launch under accelerate/deepspeed ZeRO-3; see configs).
  - 1 GPU    : 4-bit QLoRA fallback (bitsandbytes NF4) if bf16 LoRA OOMs.
Supports --smoke (tiny subset, 2 steps) for a fast wiring check on GPU.

Data: gpt_oss/data/out/trace_action_train.jsonl (chat 'messages'), eval split for metrics.

Run (multi-GPU):
  accelerate launch --config_file gpt_oss/train/accelerate_config_multih200.yaml \
    gpt_oss/train/train_lora_sft.py --base-model openai/gpt-oss-120b --out adapters/gpt_oss_120b/lever_sft_<ts>
Run (single GPU QLoRA):
  python gpt_oss/train/train_lora_sft.py --base-model openai/gpt-oss-120b --qlora --out <dir>
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

DEFAULT_DATA = os.path.join(PROJ, "gpt_oss", "data", "out", "trace_action_train.jsonl")


def load_chat_jsonl(path):
    from datasets import Dataset
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({"messages": r["messages"]})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=os.getenv("GPT_OSS_MODEL_PATH", "openai/gpt-oss-120b"))
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(PROJ, "adapters", "gpt_oss_120b",
                                                  f"lever_sft_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--qlora", action="store_true", help="4-bit NF4 (single-GPU fallback)")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 2 steps")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    cfg = vars(args).copy()
    cfg["gpu_info"] = gpu_info()
    print("GPU:", json.dumps(cfg["gpu_info"]))

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16, "use_cache": False}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    ds = load_chat_jsonl(args.data)
    if args.smoke:
        ds = ds.select(range(min(4, len(ds))))

    sft_cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=(0.0 if args.smoke else args.epochs),
        max_steps=(2 if args.smoke else -1),
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=1, save_strategy="epoch", bf16=True,
        gradient_checkpointing=True, max_seq_length=args.max_seq_len,
        report_to=(["wandb"] if os.getenv("WANDB_API_KEY") else []),
        packing=False)

    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out)            # adapter only (PEFT)
    tok.save_pretrained(args.out)

    ds_hash = write_provenance(args.out, args.base_model, args.data, cfg)
    print(f"\nadapter saved -> {args.out}\ndataset_hash {ds_hash}")
    print("Next: rollout + eval the adapter:\n"
          f"  python gpt_oss/rollout/rollout_adapter.py --adapter {args.out} ...\n"
          f"  python gpt_oss/eval/eval_adapter.py --adapter {args.out}")


if __name__ == "__main__":
    main()
