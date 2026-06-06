#!/usr/bin/env python3
"""Evaluate a LawBench LoRA adapter: predict charges on the test split, score top-1 vs private labels.

Loads base+adapter locally (PEFT) OR queries a served endpoint (--endpoint). Writes a submission.csv
and runs the official LawBench evaluate.py if available; also prints top-1 accuracy directly.
"""

import argparse
import csv
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAW = os.path.join(PROJ, "baselines", "vendor", "sia", "sia", "tasks", "lawbench")
TEST_PUB = os.path.join(LAW, "data", "public", "test.csv")
TEST_PRIV = os.path.join(LAW, "data", "private", "test.csv")

SYS = ("You are a Chinese legal expert. Given the facts (事实) of a criminal case, predict the single "
       "criminal charge (罪名) the court convicted the defendant of. Reply with ONLY the charge label.")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-model", default=os.getenv("GPT_OSS_MODEL_PATH", "openai/gpt-oss-120b"))
    ap.add_argument("--endpoint", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    test = read_csv(TEST_PUB)
    if args.limit:
        test = test[:args.limit]

    if args.endpoint:
        sys.path.insert(0, os.path.join(PROJ, "gpt_oss"))
        from client import make_client, chat
        client = make_client(base_url=args.base_url)
        def ask(text):
            return chat([{"role": "system", "content": SYS}, {"role": "user", "content": text[:6000]}],
                        model=args.model, base_url=args.base_url, client=client, max_tokens=32)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(args.base_model)
        m = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
        m = PeftModel.from_pretrained(m, args.adapter); m.eval()
        def ask(text):
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": text[:6000]}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
            out = m.generate(ids, max_new_tokens=32, do_sample=False)
            return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    preds = {}
    for r in test:
        rid = r.get("id")
        label = ask(r.get("text", "")).strip().splitlines()[0].strip() if r.get("text") else ""
        preds[rid] = label

    sub_path = os.path.join(os.path.dirname(args.adapter.rstrip("/")), "lawbench_submission.csv")
    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for rid, lab in preds.items():
            w.writerow([rid, lab])

    acc = None
    if os.path.exists(TEST_PRIV):
        priv = {r["id"]: (r.get("label") or r.get("罪名")) for r in read_csv(TEST_PRIV)}
        n = sum(1 for rid in preds if rid in priv)
        correct = sum(1 for rid, lab in preds.items() if priv.get(rid) == lab)
        acc = correct / n if n else 0.0
    res = {"adapter": args.adapter, "n": len(preds), "top1_accuracy": acc,
           "reduced_split": bool(args.limit)}
    out = os.path.join(os.path.dirname(args.adapter.rstrip("/")), "lawbench_eval.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    if args.limit:
        print("NOTE: --limit used -> REDUCED LawBench; do NOT compare to the paper headline.")


if __name__ == "__main__":
    main()
