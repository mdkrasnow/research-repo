#!/usr/bin/env python3
"""Smoke test: query gpt-oss-120b with ONE lever trace and require strict JSON.

Two modes (auto-detected):
  1. OpenAI-compatible endpoint  (GPT_OSS_BASE_URL + key + GPT_OSS_MODEL)   [default]
  2. Local HF/vLLM              (--local, GPT_OSS_MODEL_PATH or HF id)

Usage:
  python gpt_oss/smoke_infer.py --model gpt-oss-120b
  python gpt_oss/smoke_infer.py --local --model openai/gpt-oss-120b
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever_io import build_messages, parse_action  # noqa: E402

SAMPLE_TRACE = (
    "FAILED-RUN TRACE (deployed harness + model under test)\n"
    "- clean prediction MSE: 0.0002 (predicts clean examples well)\n"
    "- negative-control MSE: 0.06 (SOLVES the broken-symmetry control (suspicious))\n"
    "- learned-action composition error: 5.5 (group law violated)\n"
    "- shortcut-cheat signature present: True\n"
    "- ORACLE SANDWICH: a known-good reference model scores 0.001 under the deployed harness "
    "-> harness ACCEPTS a known-good model (harness appears valid).\n"
)


def run_endpoint(model, base_url):
    from client import chat, make_client
    client = make_client(base_url=base_url)
    return chat(build_messages(SAMPLE_TRACE), model=model, base_url=base_url, client=client)


def run_local(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    msgs = build_messages(SAMPLE_TRACE)
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=128, do_sample=False)
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("GPT_OSS_MODEL", "gpt-oss-120b"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()

    if args.local:
        mp = os.getenv("GPT_OSS_MODEL_PATH", args.model)
        print(f"[local] loading {mp}")
        raw = run_local(mp)
    else:
        print(f"[endpoint] model={args.model} base_url={args.base_url or os.getenv('GPT_OSS_BASE_URL')}")
        raw = run_endpoint(args.model, args.base_url)

    print("\n--- raw response ---\n" + str(raw))
    action, reason, valid = parse_action(raw)
    print("\n--- parsed ---")
    print(json.dumps({"action": action, "reason": reason, "valid_json": valid}, indent=2))
    ok = action in ("H", "W", "H_THEN_W", "PROMOTE", "KILL")
    print(f"\nSMOKE {'PASS' if ok else 'FAIL'} (expected H or H_THEN_W for this shortcut trace)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
