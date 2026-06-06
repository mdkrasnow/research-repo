#!/usr/bin/env python3
"""Summarize an official SIA run directory (per-generation target_agent.py / results.json /
improvement.md). Writes results/official_sia_custom_summary.md.

SIA writes runs under its own run dir; pass --run-dir explicitly or --run-id to search common spots.
"""

import argparse
import glob
import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_run_dir(run_id, run_dir):
    if run_dir:
        return run_dir
    cands = []
    for root in [os.path.join(PROJ, "runs"), os.path.join(PROJ, "baselines", "vendor", "sia"),
                 os.path.expanduser("~/.sia"), "/tmp"]:
        cands += glob.glob(os.path.join(root, "**", f"*{run_id}*"), recursive=True)
    dirs = [c for c in cands if os.path.isdir(c)]
    return dirs[0] if dirs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    rd = find_run_dir(args.run_id, args.run_dir)
    if not rd:
        raise SystemExit(f"run dir not found for run-id={args.run_id}; pass --run-dir")

    gens = sorted(glob.glob(os.path.join(rd, "gen_*")) + glob.glob(os.path.join(rd, "**", "gen_*"),
                                                                   recursive=True))
    rows = []
    for g in gens:
        res_path = next(iter(glob.glob(os.path.join(g, "**", "results.json"), recursive=True)), None)
        metrics = {}
        if res_path:
            try:
                metrics = json.load(open(res_path))
            except Exception:
                pass
        rows.append((os.path.basename(g.rstrip("/")), metrics))

    md = [f"# Official SIA-H run summary — {args.run_id or rd}", "",
          "Public SIA exposes the HARNESS loop only (no weight updates). This is a harness-loop "
          "baseline on the SIA-Lever custom task.", "",
          f"Run dir: `{rd}`", "",
          "| Generation | lever_accuracy | mean_regret | invalid_json_rate |",
          "|---|---|---|---|"]
    for name, m in rows:
        md.append(f"| {name} | {m.get('lever_accuracy','—')} | {m.get('mean_regret','—')} "
                  f"| {m.get('invalid_json_rate','—')} |")
    out = os.path.join(PROJ, "results", "official_sia_custom_summary.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
