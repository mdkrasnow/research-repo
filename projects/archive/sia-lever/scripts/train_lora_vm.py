"""VM-side LoRA trainer for gpt-oss-120b — trains on the MXFP4 base already cached on the H200,
so no Token Factory round-trip and no 235GB bf16 base (which would need 8x H200).

QLoRA-style: the MXFP4 base (~63GB) stays frozen; only the LoRA adapters (q/k/v/o_proj, bf16)
train. Fits one H200. Writes a PEFT adapter dir that vLLM can hot-load via /v1/load_lora_adapter.

Run ON the VM (inside ~/venv with transformers+peft+trl+datasets installed):
  python train_lora_vm.py --train ~/train.jsonl --out ~/adapter_v2 \
     --epochs 8 --lr 3e-5 --lora-r 32

The train jsonl is OpenAI chat format: one {"messages":[...]} per line (system/user/assistant).
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="openai/gpt-oss-120b")
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    rows = [json.loads(l) for l in open(args.train) if l.strip()]
    print(f"[train] {len(rows)} examples from {args.train}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fmt(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False,
                                                add_generation_prompt=False)}
    ds = Dataset.from_list(rows).map(fmt, remove_columns=["messages"])

    print(f"[train] loading base {args.base} (MXFP4, frozen) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, logging_steps=1, save_strategy="no",
        max_length=args.max_len, bf16=True, report_to=[], dataset_text_field="text")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
    n_steps = (len(rows) // (args.batch_size * args.grad_accum)) * args.epochs
    print(f"[train] ~{int(n_steps)} optimizer steps (epochs={args.epochs}, lr={args.lr})")
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[train] adapter saved -> {args.out}")
    # quick provenance
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump({"base": args.base, "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r,
                   "n_examples": len(rows), "batch_size": args.batch_size,
                   "grad_accum": args.grad_accum}, f, indent=2)


if __name__ == "__main__":
    main()
