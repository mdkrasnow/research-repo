"""Validate a HARD SFT train/eval pair: FAIL (exit 1) on class under-representation or train/eval
leakage. Run before any retrain so a bad dataset can't silently produce a misleading result.

Checks:
  1. No episode_id overlap between train and eval (leakage).
  2. Each label (over the active levers H / W / H_THEN_W present in train) has >= --min-frac of train.
  3. Train is non-empty; eval is non-empty.

Usage:
  python gpt_oss/data/validate_dataset.py \
     --train gpt_oss/data/out/hard_v2/trace_action_train.jsonl \
     --eval  gpt_oss/data/out/hard/trace_action_eval.jsonl
"""

import argparse
import collections
import json
import os
import sys


def _rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _label(row):
    # SFT row stores best_action; fall back to parsing assistant message
    if "best_action" in row:
        return row["best_action"]
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            try:
                return json.loads(m["content"]).get("action")
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--min-frac", type=float, default=0.15, help="min fraction of train per present label")
    args = ap.parse_args()

    train, ev = _rows(args.train), _rows(args.eval)
    fails = []

    if not train:
        fails.append("train is EMPTY")
    if not ev:
        fails.append("eval is EMPTY")

    tr_ids = {r["episode_id"] for r in train}
    ev_ids = {r["episode_id"] for r in ev}
    overlap = tr_ids & ev_ids
    if overlap:
        fails.append(f"LEAKAGE: {len(overlap)} episode_ids in BOTH train and eval, e.g. {sorted(overlap)[:3]}")

    labels = collections.Counter(_label(r) for r in train)
    n = sum(labels.values()) or 1
    for lab, c in labels.items():
        if lab is None:
            fails.append("train has rows with unparseable label")
            continue
        if c / n < args.min_frac:
            fails.append(f"label '{lab}' under-represented: {c}/{n} = {c/n:.2f} < {args.min_frac}")

    print(f"train n={len(train)} labels={dict(labels)}  | eval n={len(ev)}")
    if fails:
        print("VALIDATION FAILED:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("VALIDATION OK: no leakage, all present labels >= min-frac, non-empty.")


if __name__ == "__main__":
    main()
