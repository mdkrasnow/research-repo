#!/usr/bin/env python3
"""Export per-generation predictions from an official SIA-H LawBench run (submission.csv files),
so they can be inspected or distilled. Writes paper_benchmarks/lawbench/exported_<run>.jsonl.
"""

import argparse
import csv
import glob
import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    roots = [args.run_dir] if args.run_dir else [
        os.path.join(PROJ, "runs"), os.path.join(PROJ, "baselines", "vendor", "sia"),
        os.path.expanduser("~/.sia"), "/tmp"]
    subs = []
    for root in roots:
        if root:
            subs += glob.glob(os.path.join(root, "**", f"*{args.run_id}*", "**", "submission.csv"),
                              recursive=True)
    out_rows = []
    for s in sorted(set(subs)):
        gen = os.path.basename(os.path.dirname(s))
        with open(s, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out_rows.append({"gen": gen, "id": r.get("id"), "label": r.get("label")})

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"exported_{args.run_id}.jsonl")
    with open(out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"exported {len(out_rows)} predictions from {len(set(subs))} submission files -> {out}")
    if not subs:
        print("(no submission.csv found — check --run-dir / run-id)")


if __name__ == "__main__":
    main()
