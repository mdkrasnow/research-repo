"""Build SFT dataset (trace -> best-action JSON) from the measured cache.

Reads gpt_oss/data/out/action_outcome_cache.jsonl (real reruns).
Writes trace_action_train.jsonl / trace_action_eval.jsonl, split BY SEED.

Each row: chat messages (system/user/assistant) + episode_id + best_action + reward_by_action.
The assistant target is the best measured action; reason is templated (kept short, JSON-strict).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lever_io import build_messages, action_json, load_cache, cost_adjusted_best, PROJ  # noqa: E402

REASON = {
    "H": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.",
    "W": "Harness is valid and the model is honest but cannot predict; train weights.",
    "H_THEN_W": "Weak harness passed a shortcut-cheating model; add structural checks, then retrain.",
}


def to_sft_row(ep):
    msgs = build_messages(ep["trace_text"])
    # cost-adjusted best over the three active levers (matches the pre-registered correct lever),
    # NOT the raw argmax stored in the cache (which can pick H_THEN_W over H on a 0.001 margin).
    best = cost_adjusted_best(ep["reward_by_action"])
    target = action_json(best, REASON.get(best, "Best measured intervention for this trace."))
    return {
        "messages": msgs + [{"role": "assistant", "content": target}],
        "episode_id": ep["episode_id"],
        "mode": ep["mode"],
        "seed": ep["seed"],
        "best_action": best,
        "reward_by_action": ep["reward_by_action"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--eval-seeds", type=int, default=3,
                    help="highest N seed indices held out for eval")
    ap.add_argument("--out", default=os.path.join(PROJ, "gpt_oss", "data", "out"))
    args = ap.parse_args()

    rows = load_cache(args.cache)
    seeds = sorted({r["seed"] for r in rows})
    eval_set = set(seeds[-args.eval_seeds:]) if args.eval_seeds else set()
    train = [to_sft_row(r) for r in rows if r["seed"] not in eval_set]
    ev = [to_sft_row(r) for r in rows if r["seed"] in eval_set]

    os.makedirs(args.out, exist_ok=True)
    for name, data in [("trace_action_train.jsonl", train), ("trace_action_eval.jsonl", ev)]:
        with open(os.path.join(args.out, name), "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
    print(f"SFT: {len(train)} train / {len(ev)} eval (eval seeds {sorted(eval_set)})")
    _validate(train + ev)


def _validate(rows):
    from lever_io import VALID_ACTIONS, cost_adjusted_best
    ids = set()
    for r in rows:
        assert r["episode_id"] not in ids, f"dup episode_id {r['episode_id']}"
        ids.add(r["episode_id"])
        assert r["best_action"] in VALID_ACTIONS, f"bad action {r['best_action']}"
        # label must equal the cost-adjusted best over the three active levers
        assert r["best_action"] == cost_adjusted_best(r["reward_by_action"]), "label != cost-adjusted best"
    print(f"validation OK: {len(rows)} rows, unique ids, label=cost-adjusted best")


if __name__ == "__main__":
    main()
