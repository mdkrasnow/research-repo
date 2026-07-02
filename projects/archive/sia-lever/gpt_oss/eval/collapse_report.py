"""Anti-degeneracy eval gate for the lever selector.

The 20-epoch HARD LoRA "beat" the tiny floor only by predicting H for ALL 24 eval episodes
(gold H=11/24=0.458). Accuracy alone cannot tell that apart from real attribution. This report
brackets a model between the BASE and the CONSTANT-policy baselines and flags collapse explicitly.

A model is a REAL WIN only if it:
  (a) beats base on lever_accuracy, AND
  (b) beats every constant-policy baseline (always-H / always-W / always-H_THEN_W) on accuracy, AND
  (c) does not materially worsen mean_regret vs base (tolerance --regret-tol, default 0.02), AND
  (d) is NOT collapsed (no single action >80% of predictions).

Usage:
  python gpt_oss/eval/collapse_report.py \
      --rollouts 'results/gpt_oss/lora_20ep_rollouts_*.jsonl' \
      --base-rollouts 'results/gpt_oss/base_20ep_rollouts_*.jsonl' \
      --cache gpt_oss/data/out/hard_cache.jsonl --tag lora_20ep
Exit code 0 always; the verdict is in the printed JSON + markdown (REAL_WIN | COLLAPSE | NO_WIN).
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

COLLAPSE_FRAC = 0.80   # one action predicted for >80% of episodes => collapsed


def _load(pattern):
    paths = sorted(glob.glob(pattern))
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def _metrics(actions_by_id, cache_by_id):
    """actions_by_id: {episode_id: action}. Score vs measured gold."""
    n = correct = 0
    regrets, dist, per_label, per_mode = [], {}, {}, {}
    for eid, act in actions_by_id.items():
        ep = cache_by_id[eid]
        rba = ep["reward_by_action"]
        gold = cost_adjusted_best(rba)
        a = act if act in VALID_ACTIONS else "KILL"
        regrets.append(regret_of_action(a, rba))
        dist[a] = dist.get(a, 0) + 1
        pl = per_label.setdefault(gold, {"c": 0, "n": 0})
        pl["n"] += 1; pl["c"] += int(a == gold)
        pm = per_mode.setdefault(ep["mode"], {"c": 0, "n": 0})
        pm["n"] += 1; pm["c"] += int(a == gold)
        correct += int(a == gold); n += 1
    return {
        "n": n,
        "lever_accuracy": correct / n if n else 0.0,
        "mean_regret": st.mean(regrets) if regrets else 0.0,
        "max_regret": max(regrets) if regrets else 0.0,
        "action_distribution": dist,
        "per_label_accuracy": {k: v["c"] / v["n"] for k, v in per_label.items()},
        "per_mode_accuracy": {k: v["c"] / v["n"] for k, v in per_mode.items()},
    }


def _constant_baselines(eval_ids, cache_by_id):
    out = {}
    for const in ("H", "W", "H_THEN_W"):
        m = _metrics({eid: const for eid in eval_ids}, cache_by_id)
        out[const] = {"accuracy": m["lever_accuracy"], "mean_regret": m["mean_regret"]}
    return out


def _is_collapsed(dist, n):
    if not n:
        return False, None
    top_a = max(dist, key=dist.get)
    frac = dist[top_a] / n
    return (frac > COLLAPSE_FRAC), {"action": top_a, "frac": round(frac, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--base-rollouts", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--regret-tol", type=float, default=0.02,
                    help="max mean_regret increase vs base still allowed for a win")
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "gpt_oss"))
    args = ap.parse_args()

    cache = load_cache(args.cache)
    cache_by_id = {r["episode_id"]: r for r in cache}
    roll = _load(args.rollouts)
    actions = {r["episode_id"]: r.get("action") for r in roll}
    eval_ids = list(actions.keys())

    m = _metrics(actions, cache_by_id)
    gold_dist = {}
    for eid in eval_ids:
        g = cost_adjusted_best(cache_by_id[eid]["reward_by_action"])
        gold_dist[g] = gold_dist.get(g, 0) + 1
    consts = _constant_baselines(eval_ids, cache_by_id)
    collapsed, collapse_info = _is_collapsed(m["action_distribution"], m["n"])

    base = None
    if args.base_rollouts:
        broll = _load(args.base_rollouts)
        base = _metrics({r["episode_id"]: r.get("action") for r in broll}, cache_by_id)

    best_const_acc = max(c["accuracy"] for c in consts.values())
    beats_base = base is not None and m["lever_accuracy"] > base["lever_accuracy"]
    beats_consts = m["lever_accuracy"] > best_const_acc
    regret_ok = base is not None and (m["mean_regret"] <= base["mean_regret"] + args.regret_tol)
    real_win = bool(beats_base and beats_consts and regret_ok and not collapsed)
    verdict = "REAL_WIN" if real_win else ("COLLAPSE" if collapsed else "NO_WIN")

    rep = {
        "tag": args.tag, "verdict": verdict, "collapsed": collapsed, "collapse_info": collapse_info,
        "model": {k: m[k] for k in ("n", "lever_accuracy", "mean_regret", "max_regret",
                                    "action_distribution", "per_label_accuracy", "per_mode_accuracy")},
        "gold_distribution": gold_dist,
        "constant_baselines": consts, "best_constant_accuracy": best_const_acc,
        "base": ({"lever_accuracy": base["lever_accuracy"], "mean_regret": base["mean_regret"]}
                 if base else None),
        "checks": {"beats_base": beats_base, "beats_constants": beats_consts,
                   "regret_not_worse": regret_ok, "not_collapsed": not collapsed},
    }
    os.makedirs(args.out, exist_ok=True)
    jpath = os.path.join(args.out, f"collapse_report_{args.tag}.json")
    with open(jpath, "w") as f:
        json.dump(rep, f, indent=2)

    print(f"\n=== COLLAPSE REPORT [{args.tag}] -> {verdict} ===")
    print(f"accuracy {m['lever_accuracy']:.3f}  regret {m['mean_regret']:.3f}  "
          f"(base acc {base['lever_accuracy']:.3f} / regret {base['mean_regret']:.3f})"
          if base else f"accuracy {m['lever_accuracy']:.3f}  regret {m['mean_regret']:.3f}")
    print(f"action_dist {m['action_distribution']}  gold {gold_dist}")
    print(f"constant baselines (acc): " + ", ".join(f"{k}={v['accuracy']:.3f}" for k, v in consts.items())
          + f"  -> best constant {best_const_acc:.3f}")
    if collapsed:
        print(f"** COLLAPSE: predicts {collapse_info['action']} for {collapse_info['frac']*100:.0f}% of episodes **")
    print(f"checks: {rep['checks']}")
    print(f"saved {jpath}")


if __name__ == "__main__":
    main()
