"""Evaluate an adapter rollout vs the base rollout on the lever task and report deltas.

Reuses eval_selector.score for both, writes results/gpt_oss/adapter_eval_<ts>.{json,md} with
per-mode accuracy and base->adapter improvement on lever_accuracy / mean_regret / invalid_json.

Usage:
  python gpt_oss/eval/eval_adapter.py --adapter-rollouts results/gpt_oss/sft_rollouts_*.jsonl \
                                      --base-rollouts results/gpt_oss/base_rollouts_*.jsonl --tag sft
"""

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from lever_io import load_cache  # noqa: E402
sys.path.insert(0, HERE)
from eval_selector import load_rollouts, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-rollouts", required=True)
    ap.add_argument("--base-rollouts", default=None)
    ap.add_argument("--tag", default="sft")
    args = ap.parse_args()

    cache = load_cache()
    adapter_rows, _ = load_rollouts(args.adapter_rollouts)
    adapter = score(adapter_rows, cache)
    base = None
    if args.base_rollouts and glob.glob(args.base_rollouts):
        base_rows, _ = load_rollouts(args.base_rollouts)
        base = score(base_rows, cache)

    res = {"tag": args.tag, "adapter": adapter, "base": base}
    if base:
        res["delta"] = {
            "lever_accuracy": adapter["lever_accuracy"] - base["lever_accuracy"],
            "mean_regret": adapter["mean_regret"] - base["mean_regret"],
            "invalid_json_rate": adapter["invalid_json_rate"] - base["invalid_json_rate"],
        }

    d = os.path.join(PROJ, "results", "gpt_oss")
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with open(os.path.join(d, f"adapter_eval_{stamp}.json"), "w") as f:
        json.dump(res, f, indent=2)

    md = [f"# Adapter eval — {args.tag} ({stamp})", "",
          "| metric | base gpt-oss | gpt-oss+LoRA | delta |", "|---|---|---|---|"]
    for k in ["lever_accuracy", "mean_regret", "max_regret", "invalid_json_rate"]:
        b = f"{base[k]:.3f}" if base else "—"
        delta = f"{adapter[k]-base[k]:+.3f}" if base and k in base else "—"
        md.append(f"| {k} | {b} | {adapter[k]:.3f} | {delta} |")
    md += ["", f"Adapter per-mode accuracy: {json.dumps(adapter['per_mode_accuracy'])}"]
    with open(os.path.join(d, f"adapter_eval_{stamp}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nsaved adapter_eval_{stamp}.{{json,md}}")


if __name__ == "__main__":
    main()
