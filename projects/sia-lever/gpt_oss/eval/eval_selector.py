"""Score a lever-selector rollout file against the measured outcome cache.

Computes (all from REAL measured outcomes via lever_io.regret_of_action):
  lever_accuracy        fraction matching the cost-adjusted correct lever
  mean_regret/max_regret real-rerun regret of the chosen action
  invalid_json_rate
  action_distribution
  per_mode_accuracy     shortcut_leak / model_prior_gap / bad_verifier
  n_examples

Usage:
  python gpt_oss/eval/eval_selector.py --rollouts results/gpt_oss/base_rollouts_*.jsonl --tag base
"""

import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lever_io import (load_cache, cost_adjusted_best, regret_of_action,  # noqa: E402
                      VALID_ACTIONS, PROJ)


def load_rollouts(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no rollouts match {pattern}")
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows, paths


def score(rollouts, cache):
    by_id = {r["episode_id"]: r for r in cache}
    n = len(rollouts)
    correct = 0
    regrets = []
    invalid = 0
    dist = {a: 0 for a in VALID_ACTIONS}
    dist["INVALID"] = 0
    per_mode = {}
    for r in rollouts:
        ep = by_id.get(r["episode_id"])
        if ep is None:
            raise KeyError(f"episode {r['episode_id']} not in cache — cannot score without measured outcome")
        rba = ep["reward_by_action"]
        gold = cost_adjusted_best(rba)
        act = r.get("action")
        if not r.get("valid_json", False):
            invalid += 1
        if act in VALID_ACTIONS:
            dist[act] += 1
        else:
            dist["INVALID"] += 1
            act = "KILL"  # unparseable -> worst-case action for regret (no real intervention)
        regrets.append(regret_of_action(act, rba))
        pm = per_mode.setdefault(ep["mode"], {"correct": 0, "n": 0})
        pm["n"] += 1
        pm["correct"] += int(act == gold)
        correct += int(act == gold)
    return {
        "n_examples": n,
        "lever_accuracy": correct / n if n else 0.0,
        "mean_regret": st.mean(regrets) if regrets else 0.0,
        "max_regret": max(regrets) if regrets else 0.0,
        "invalid_json_rate": invalid / n if n else 0.0,
        "action_distribution": dist,
        "per_mode_accuracy": {m: v["correct"] / v["n"] for m, v in per_mode.items()},
        "per_mode_counts": per_mode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="glob for rollout jsonl")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "gpt_oss"))
    args = ap.parse_args()

    rollouts, paths = load_rollouts(args.rollouts)
    cache = load_cache()
    res = score(rollouts, cache)
    res["rollout_files"] = paths
    res["tag"] = args.tag

    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, f"{args.tag}_eval.json")
    with open(outp, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in
                      ["n_examples", "lever_accuracy", "mean_regret", "max_regret",
                       "invalid_json_rate", "per_mode_accuracy"]}, indent=2))
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
