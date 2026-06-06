"""Generate the SIA custom-task data files from the measured cache.

Splits gpt_oss/data/out/action_outcome_cache.jsonl into:
  data/public/traces_public.jsonl   episode_id + mode + observable_trace + trace_text  (NO labels)
  data/private/traces_hidden.jsonl  episode_id (the eval subset the agent is scored on)
  data/private/measured_outcomes.jsonl  episode_id + reward_by_action + correct_action  (ground truth)

Public = what the target agent sees. Private = ground truth for evaluate.py. No fabrication: all
rewards come from the real-rerun cache.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "gpt_oss"))
from lever_io import load_cache, cost_adjusted_best  # noqa: E402

EVAL_SEEDS = 3  # held-out eval subset the agent is scored on (highest seed indices)


def main():
    cache = load_cache()
    seeds = sorted({r["seed"] for r in cache})
    eval_set = set(seeds[-EVAL_SEEDS:])

    pub = os.path.join(HERE, "data", "public")
    priv = os.path.join(HERE, "data", "private")
    os.makedirs(pub, exist_ok=True)
    os.makedirs(priv, exist_ok=True)

    with open(os.path.join(pub, "traces_public.jsonl"), "w") as f:
        for r in cache:
            f.write(json.dumps({
                "episode_id": r["episode_id"], "mode": r["mode"],
                "observable_trace": r["observable_trace"], "trace_text": r["trace_text"],
            }) + "\n")

    n_hidden = 0
    with open(os.path.join(priv, "traces_hidden.jsonl"), "w") as fh, \
         open(os.path.join(priv, "measured_outcomes.jsonl"), "w") as fo:
        for r in cache:
            if r["seed"] not in eval_set:
                continue
            n_hidden += 1
            fh.write(json.dumps({"episode_id": r["episode_id"], "mode": r["mode"]}) + "\n")
            fo.write(json.dumps({
                "episode_id": r["episode_id"], "mode": r["mode"],
                "reward_by_action": r["reward_by_action"],
                "correct_action": cost_adjusted_best(r["reward_by_action"]),
            }) + "\n")

    with open(os.path.join(pub, "sample_submission.json"), "w") as f:
        json.dump({"predictions": [
            {"episode_id": cache[0]["episode_id"], "action": "H_THEN_W",
             "reason": "example: clean prediction succeeds but neg-control also succeeds -> fix harness then train."}
        ]}, f, indent=2)

    print(f"public traces: {len(cache)}; hidden eval episodes: {n_hidden} (seeds {sorted(eval_set)})")


if __name__ == "__main__":
    main()
