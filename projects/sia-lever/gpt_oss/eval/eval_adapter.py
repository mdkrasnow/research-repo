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
from lever_io import load_cache, cost_adjusted_best  # noqa: E402
sys.path.insert(0, HERE)
from eval_selector import load_rollouts, score  # noqa: E402
import plots  # noqa: E402


def episode_diff(base_rows, adapter_rows, cache):
    """Categorize each episode: fixed (base wrong→adapter right), regressed, both_right, both_wrong."""
    by_id = {r["episode_id"]: r for r in cache}
    base_a = {r["episode_id"]: r.get("action") for r in base_rows}
    adap_a = {r["episode_id"]: r.get("action") for r in adapter_rows}
    cats = {"fixed": [], "regressed": [], "both_right": [], "both_wrong": []}
    for eid, ep in by_id.items():
        if eid not in adap_a:
            continue
        gold = cost_adjusted_best(ep["reward_by_action"])
        b_ok = base_a.get(eid) == gold
        a_ok = adap_a.get(eid) == gold
        key = ("both_right" if b_ok and a_ok else "both_wrong" if not b_ok and not a_ok
               else "fixed" if a_ok else "regressed")
        cats[key].append({"episode_id": eid, "mode": ep["mode"], "correct": gold,
                          "base": base_a.get(eid), "adapter": adap_a.get(eid)})
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-rollouts", required=True)
    ap.add_argument("--base-rollouts", default=None)
    ap.add_argument("--tag", default="sft")
    ap.add_argument("--out", default=None)
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

    d = args.out or os.path.join(PROJ, "results", "gpt_oss")
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # --- headline base-vs-LoRA figure (the demo money shot) ---
    fig_path = os.path.join(d, f"adapter_eval_{stamp}.png")
    plots.base_vs_adapter_fig(base, adapter, fig_path, tag=args.tag.upper())
    plots.per_mode_bar(adapter["per_mode_accuracy"],
                       f"{args.tag}: per-mode accuracy (vs base)",
                       os.path.join(d, f"adapter_per_mode_{stamp}.png"),
                       ref=(base["per_mode_accuracy"] if base else None))

    # --- fixed/regressed episode diff (debug what the adapter changed) ---
    diff = None
    if base and args.base_rollouts and glob.glob(args.base_rollouts):
        base_rows, _ = load_rollouts(args.base_rollouts)
        diff = episode_diff(base_rows, adapter_rows, cache)
        res["episode_diff_counts"] = {k: len(v) for k, v in diff.items()}

    with open(os.path.join(d, f"adapter_eval_{stamp}.json"), "w") as f:
        json.dump(res, f, indent=2)

    md = [f"# Adapter eval — {args.tag} ({stamp})", "",
          f"Figure: `adapter_eval_{stamp}.png` (base vs +{args.tag.upper()})", "",
          "| metric | base gpt-oss | gpt-oss+LoRA | delta |", "|---|---|---|---|"]
    for k in ["lever_accuracy", "mean_regret", "max_regret", "invalid_json_rate"]:
        b = f"{base[k]:.3f}" if base else "—"
        delta = f"{adapter[k]-base[k]:+.3f}" if base and k in base else "—"
        md.append(f"| {k} | {b} | {adapter[k]:.3f} | {delta} |")
    md += ["", f"Adapter per-mode accuracy: {json.dumps(adapter['per_mode_accuracy'])}"]
    if diff:
        md += ["", "## What LoRA changed vs base",
               f"- **fixed** (base wrong → LoRA right): {len(diff['fixed'])}",
               f"- regressed (base right → LoRA wrong): {len(diff['regressed'])}",
               f"- both right: {len(diff['both_right'])} · both wrong: {len(diff['both_wrong'])}"]
        for cat in ("fixed", "regressed"):
            for e in diff[cat]:
                md.append(f"  - [{cat}] {e['episode_id']} ({e['mode']}): "
                          f"base=`{e['base']}` lora=`{e['adapter']}` correct=`{e['correct']}`")
    with open(os.path.join(d, f"adapter_eval_{stamp}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nsaved adapter_eval_{stamp}.{{json,md,png}} + per_mode png")


if __name__ == "__main__":
    main()
