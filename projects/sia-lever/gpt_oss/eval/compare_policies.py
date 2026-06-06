"""Compare lever policies on the measured eval cache (real-rerun regret). CPU-only for the
rule/fixed/paper-style policies; gpt-oss base/LoRA columns are filled from rollout files if given.

Policies:
  H_only, W_only, alternating            fixed (ignore trace)
  plateau_then_w                         paper-style scheduler (re-measures after H)
  oracle_sandwich_rule                   our deterministic lever-aware rule (label-free)
  base_gpt_oss / gpt_oss_lora            learned selectors (from --base-rollouts/--adapter-rollouts)
  oracle_best                            upper bound (always cost-adjusted best)

Metrics per policy: lever_accuracy, mean_regret, max_regret, w_calls, invalid_json_rate.
Writes results/final_comparison.csv + .md and plots/final_comparison.png.

Usage:
  python gpt_oss/eval/compare_policies.py                       # CPU policies only
  python gpt_oss/eval/compare_policies.py --base-rollouts results/gpt_oss/base_rollouts_*.jsonl \
                                          --adapter-rollouts results/gpt_oss/sft_rollouts_*.jsonl
"""

import argparse
import csv
import glob
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lever_io import (load_cache, cost_adjusted_best, regret_of_action,  # noqa: E402
                      VALID_ACTIONS, PROJ)
sys.path.insert(0, os.path.join(PROJ, "methods"))
sys.path.insert(0, os.path.join(PROJ, "baselines", "paper_style"))
import oracle_sandwich_selector  # noqa: E402
import plateau_then_w  # noqa: E402
import fixed_policy_baselines as fixed  # noqa: E402
import learned_selector  # noqa: E402

W_ACTIONS = {"W", "H_THEN_W"}


def eval_episodes(cache, eval_seeds):
    seeds = sorted({r["seed"] for r in cache})
    keep = set(seeds[-eval_seeds:]) if eval_seeds else set(seeds)
    return [r for r in cache if r["seed"] in keep]


def score_policy(name, action_fn, episodes):
    """action_fn(ep, idx) -> action token (may be None for unparseable)."""
    regrets, correct, w_calls, invalid = [], 0, 0, 0
    for idx, ep in enumerate(episodes):
        rba = ep["reward_by_action"]
        gold = cost_adjusted_best(rba)
        act = action_fn(ep, idx)
        if act not in VALID_ACTIONS:
            invalid += 1
            act = "KILL"  # unparseable -> worst-case
        if act in W_ACTIONS:
            w_calls += 1
        correct += int(act == gold)
        regrets.append(regret_of_action(act, rba))
    n = len(episodes)
    return {"policy": name, "lever_accuracy": correct / n, "mean_regret": st.mean(regrets),
            "max_regret": max(regrets), "w_calls": w_calls, "invalid_json_rate": invalid / n,
            "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--all", action="store_true", help="score on all seeds, not just eval split")
    ap.add_argument("--base-rollouts", default=None)
    ap.add_argument("--adapter-rollouts", default=None)
    ap.add_argument("--out-dir", default=None, help="override output dir (default: results/ + plots/)")
    ap.add_argument("--title-tag", default="", help="prefix for plot title (e.g. PREVIEW)")
    args = ap.parse_args()
    res_dir = args.out_dir or os.path.join(PROJ, "results")
    plot_dir = args.out_dir or os.path.join(PROJ, "plots")

    cache = load_cache()
    episodes = eval_episodes(cache, 0 if args.all else args.eval_seeds)

    policies = []
    policies.append(score_policy("H_only", lambda ep, i: fixed.h_only(ep, i), episodes))
    policies.append(score_policy("W_only", lambda ep, i: fixed.w_only(ep, i), episodes))
    policies.append(score_policy("alternating", lambda ep, i: fixed.alternating(ep, i), episodes))
    policies.append(score_policy("plateau_then_w",
                                 lambda ep, i: plateau_then_w.select_from_episode(ep), episodes))
    policies.append(score_policy("oracle_sandwich_rule",
                                 lambda ep, i: oracle_sandwich_selector.select_from_episode(ep), episodes))

    for label, path in [("base_gpt_oss", args.base_rollouts), ("gpt_oss_lora", args.adapter_rollouts)]:
        if not path:
            continue
        matches = sorted(glob.glob(path))
        if not matches:
            print(f"[warn] no rollouts for {label}: {path}")
            continue
        amap = learned_selector.from_rollouts(matches[-1])
        policies.append(score_policy(label, lambda ep, i, m=amap: m.get(ep["episode_id"]), episodes))

    policies.append(score_policy("oracle_best",
                                 lambda ep, i: cost_adjusted_best(ep["reward_by_action"]), episodes))

    # ---- write ----
    cols = ["policy", "lever_accuracy", "mean_regret", "max_regret", "w_calls",
            "invalid_json_rate", "n"]
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    with open(os.path.join(res_dir, "final_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in policies:
            w.writerow(p)

    md = ["# SIA-Lever-120B — policy comparison (measured eval-set regret)", "",
          f"Episodes: {policies[0]['n']} ({'all seeds' if args.all else f'eval seeds, held-out'}). "
          "Regret = best cost-adjusted measured outcome − chosen action's measured outcome (real "
          "reruns, no transition table). W calls = actions that trigger a weight update (W or H_THEN_W).",
          "",
          "| Policy | Lever Acc ↑ | Mean Regret ↓ | Max Regret ↓ | W calls | Invalid JSON |",
          "|---|---|---|---|---|---|"]
    for p in policies:
        md.append(f"| {p['policy']} | {p['lever_accuracy']:.2f} | {p['mean_regret']:.3f} "
                  f"| {p['max_regret']:.3f} | {p['w_calls']}/{p['n']} | {p['invalid_json_rate']:.2f} |")
    md += ["", "Notes:",
           "- oracle_sandwich_rule is a deterministic upper-bound diagnostic (label-free rule).",
           "- oracle_best is the unreachable ceiling (always the cost-adjusted best lever).",
           "- base_gpt_oss / gpt_oss_lora appear only when rollout files are supplied (need endpoint/GPU).",
           "- plateau_then_w is a paper-STYLE scheduler, not an exact SIA reproduction."]
    with open(os.path.join(res_dir, "final_comparison.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    _plot(policies, os.path.join(plot_dir, "final_comparison.png"), title_tag=args.title_tag)
    print(f"\nsaved final_comparison.{{csv,md}} + final_comparison.png -> {res_dir}")


def _plot(policies, path, title_tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [p["policy"] for p in policies]
    regret = [p["mean_regret"] for p in policies]
    acc = [p["lever_accuracy"] for p in policies]
    wcalls = [p["w_calls"] / max(p["n"], 1) for p in policies]
    # highlight our policies vs baselines vs oracles
    def color(p):
        if p in ("oracle_best", "oracle_sandwich_rule"):
            return "tab:green"
        if p in ("gpt_oss_lora", "base_gpt_oss"):
            return "tab:purple"
        return "tab:gray"
    cols = [color(n) for n in names]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, vals, title, xlim in [
            (ax[0], regret, "Mean regret (lower=better)", None),
            (ax[1], acc, "Lever accuracy (higher=better)", (0, 1.05)),
            (ax[2], wcalls, "W-call rate (weight updates pulled)", (0, 1.05))]:
        bars = a.barh(names, vals, color=cols)
        a.invert_yaxis(); a.set_title(title)
        if xlim:
            a.set_xlim(*xlim)
        for b, v in zip(bars, vals):
            a.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.2f}", va="center", fontsize=8)
    fig.suptitle(f"{title_tag+' ' if title_tag else ''}SIA-Lever-120B: lever-attribution policy "
                 "comparison (measured)  [green=oracle/rule, purple=gpt-oss, gray=baseline]")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
