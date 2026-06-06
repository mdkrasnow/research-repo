"""Build DPO preference pairs from the measured cache.

chosen   = best measured action JSON
rejected = a plausible WRONG action whose measured reward is strictly lower
Each pair carries measured_reward_chosen / measured_reward_rejected / regret_gap.

Validation: chosen reward >= rejected reward (with tolerance); split by seed.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lever_io import build_messages, action_json, load_cache, cost_adjusted_best, PROJ  # noqa: E402

REASON = {
    "H": "Harness rejects a known-good model; fix the evaluator, do not train on bad feedback.",
    "W": "Harness valid, model honest-but-weak; train weights.",
    "H_THEN_W": "Weak harness passed a cheater; add structural checks then retrain.",
    "NOOP": "Declare solved and move on.",
    "KILL": "Abandon the mechanism.",
}
# map measured cache keys to emitted action tokens (NOOP -> PROMOTE in the action vocabulary)
TOKEN = {"H": "H", "W": "W", "H_THEN_W": "H_THEN_W", "NOOP": "PROMOTE", "KILL": "KILL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.02, help="min reward gap to emit a pair")
    ap.add_argument("--out", default=os.path.join(PROJ, "gpt_oss", "data", "out"))
    args = ap.parse_args()

    rows = load_cache(args.cache)
    seeds = sorted({r["seed"] for r in rows})
    eval_set = set(seeds[-args.eval_seeds:]) if args.eval_seeds else set()

    train, ev = [], []
    for r in rows:
        rba = r["reward_by_action"]
        best = cost_adjusted_best(rba)   # cost-adjusted correct lever, not raw argmax
        best_r = rba[best]
        msgs = build_messages(r["trace_text"])
        chosen = action_json(TOKEN[best], REASON.get(best, ""))
        for alt, alt_r in rba.items():
            if alt == best:
                continue
            gap = best_r - alt_r
            if gap < args.tol:
                continue  # not clearly worse -> skip (avoid noisy pairs)
            pair = {
                "episode_id": r["episode_id"], "mode": r["mode"], "seed": r["seed"],
                "prompt_messages": msgs,
                "chosen": chosen,
                "rejected": action_json(TOKEN[alt], REASON.get(alt, "")),
                "measured_reward_chosen": best_r,
                "measured_reward_rejected": alt_r,
                "regret_gap": round(gap, 6),
            }
            (ev if r["seed"] in eval_set else train).append(pair)

    os.makedirs(args.out, exist_ok=True)
    for name, data in [("dpo_pairs_train.jsonl", train), ("dpo_pairs_eval.jsonl", ev)]:
        with open(os.path.join(args.out, name), "w") as f:
            for p in data:
                f.write(json.dumps(p) + "\n")
    # validate
    for p in train + ev:
        assert p["measured_reward_chosen"] >= p["measured_reward_rejected"], "chosen<rejected"
    print(f"DPO: {len(train)} train / {len(ev)} eval pairs (eval seeds {sorted(eval_set)}); "
          f"all chosen_reward >= rejected_reward")


if __name__ == "__main__":
    main()
