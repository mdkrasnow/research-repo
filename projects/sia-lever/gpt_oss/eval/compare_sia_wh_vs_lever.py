#!/usr/bin/env python3
"""Head-to-head: SIA-W+H (paper-style) vs SIA-Lever (ours) on the measured TriMul GPU-kernel cache.

This is the comparison the user asks for: the SIA paper's Feedback-Agent H-vs-W *scheduler*
(reconstructed paper-style as `plateau_then_w` — the public SIA repo ships NO W code, so an exact
reproduction is impossible; see documentation/reproduction_limits.md) versus SIA-Lever's learned /
rule lever selector, scored on the SAME real-rerun episodes.

Controls (CLAUDE.md mandates a positive + negative control bracketing every treatment):
  - oracle_best   POSITIVE control / upper bound: always the cost-adjusted best lever (regret 0).
  - W_only        NEGATIVE control: always pull W (the reflex "just retrain" floor).
  - H_only        NEGATIVE control: always pull H (the reflex "just fix harness" floor).
Read the two treatments ONLY in the band between these controls.

Treatments:
  - sia_wh_plateau   SIA-W+H paper-style scheduler (plateau_then_w).
  - sia_lever_rule   SIA-Lever deterministic kernel rule (methods/kernel_lever_rule.py).
  - sia_lever_lora   SIA-Lever learned selector, from --lever-rollouts (gpt-oss ± LoRA), if given.

Metrics per policy: lever_accuracy, mean_regret, max_regret, w_calls (paid retrains),
invalid_json_rate. Writes results/trimul_sia_wh_vs_lever.{csv,md} and plots/trimul_sia_wh_vs_lever.png.

Usage:
  python gpt_oss/eval/compare_sia_wh_vs_lever.py                              # rule + scheduler + controls
  python gpt_oss/eval/compare_sia_wh_vs_lever.py --lever-rollouts results/gpt_oss/sft_rollouts_*.jsonl
"""

import argparse
import csv
import glob
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJ, "gpt_oss"))
sys.path.insert(0, os.path.join(PROJ, "methods"))
sys.path.insert(0, os.path.join(PROJ, "baselines", "paper_style"))
from lever_io import load_cache, cost_adjusted_best, regret_of_action, VALID_ACTIONS  # noqa: E402
import kernel_lever_rule  # noqa: E402
import plateau_then_w  # noqa: E402

DEFAULT_CACHE = os.path.join(PROJ, "gpt_oss", "data", "out", "kernel_cache.jsonl")
W_ACTIONS = {"W", "H_THEN_W"}


def eval_episodes(cache, eval_seeds):
    seeds = sorted({r["seed"] for r in cache})
    keep = set(seeds[-eval_seeds:]) if eval_seeds else set(seeds)
    return [r for r in cache if r["seed"] in keep]


def score_policy(name, action_fn, episodes):
    regrets, correct, w_calls, invalid = [], 0, 0, 0
    for idx, ep in enumerate(episodes):
        rba = ep["reward_by_action"]
        gold = ep.get("correct_action") or cost_adjusted_best(rba)
        act = action_fn(ep, idx)
        if act not in VALID_ACTIONS:
            invalid += 1
            act = "KILL"   # unparseable -> worst-case
        if act in W_ACTIONS:
            w_calls += 1
        correct += int(act == gold)
        regrets.append(regret_of_action(act, rba))
    n = len(episodes)
    return {"policy": name, "lever_accuracy": correct / n, "mean_regret": st.mean(regrets),
            "max_regret": max(regrets), "w_calls": w_calls, "invalid_json_rate": invalid / n, "n": n}


def rollouts_to_map(pattern):
    import json
    out = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["episode_id"]] = r.get("action")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--eval-seeds", type=int, default=3, help="0 = score on all seeds")
    ap.add_argument("--lever-rollouts", default=None,
                    help="glob of gpt-oss rollout jsonl -> adds the learned SIA-Lever column")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cache = load_cache(args.cache)
    episodes = eval_episodes(cache, args.eval_seeds)

    policies = [
        ("oracle_best   [POS control]", lambda ep, i: cost_adjusted_best(ep["reward_by_action"])),
        ("W_only        [NEG control]", lambda ep, i: "W"),
        ("H_only        [NEG control]", lambda ep, i: "H"),
        ("sia_wh_plateau [paper-style]", lambda ep, i: plateau_then_w.select_from_episode(ep)),
        ("sia_lever_rule [ours]", lambda ep, i: kernel_lever_rule.select_from_episode(ep)),
    ]
    if args.lever_rollouts:
        amap = rollouts_to_map(args.lever_rollouts)
        policies.append(("sia_lever_lora [ours,learned]", lambda ep, i: amap.get(ep["episode_id"])))

    rows = [score_policy(name, fn, episodes) for name, fn in policies]

    res_dir = args.out_dir or os.path.join(PROJ, "results")
    plot_dir = args.out_dir or os.path.join(PROJ, "plots")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    cols = ["policy", "lever_accuracy", "mean_regret", "max_regret", "w_calls", "invalid_json_rate", "n"]
    with open(os.path.join(res_dir, "trimul_sia_wh_vs_lever.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [f"# SIA-W+H (paper-style) vs SIA-Lever — TriMul GPU-kernel task",
             f"\nEval episodes: {len(episodes)} (cache: `{os.path.relpath(args.cache, PROJ)}`)\n",
             "| policy | lever_acc | mean_regret | max_regret | w_calls | invalid_json |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['policy']} | {r['lever_accuracy']:.3f} | {r['mean_regret']:.4f} | "
                     f"{r['max_regret']:.4f} | {r['w_calls']} | {r['invalid_json_rate']:.3f} |")
    lines += [
        "\n**Read between the controls.** oracle_best = upper bound (regret 0); W_only / H_only = "
        "floors. A treatment near the floor = dead; near oracle = works.",
        "\n**Honesty:** the public SIA repo ships the harness (H) loop only — no W code — so "
        "`sia_wh_plateau` is a paper-STYLE reconstruction of the paper's H-vs-W scheduler, not an "
        "exact reproduction. SIA-Lever's edge is naming the **minimal correct lever** (lever_acc + "
        "fewer paid W retrains at equal regret), not necessarily lower regret than a competent "
        "scheduler. See documentation/reproduction_limits.md.",
    ]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(res_dir, "trimul_sia_wh_vs_lever.md"), "w") as f:
        f.write(md)
    print(md)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["policy"].split()[0] for r in rows]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        a1.bar(names, [r["lever_accuracy"] for r in rows], color="steelblue")
        a1.set_title("Lever accuracy (higher=better)"); a1.set_ylim(0, 1)
        a1.tick_params(axis="x", rotation=45)
        a2.bar(names, [r["mean_regret"] for r in rows], color="indianred")
        a2.set_title("Mean regret (lower=better)")
        a2.tick_params(axis="x", rotation=45)
        fig.suptitle("SIA-W+H (paper-style) vs SIA-Lever — TriMul GPU-kernel task")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "trimul_sia_wh_vs_lever.png"), dpi=120)
        print(f"plot -> {os.path.join(plot_dir, 'trimul_sia_wh_vs_lever.png')}")
    except Exception as e:  # noqa: BLE001
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
