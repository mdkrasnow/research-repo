#!/usr/bin/env python3
"""Generate the SIA custom-task data files for SIA-Lever-TriMul (the GPU-kernel lever task).

Reads the MEASURED kernel cache (gpt_oss/data/out/kernel_cache.jsonl, built by
experiments/trimul_task.py — add --real-latency on a GPU box) and splits it into the SIA
custom-task contract:

  data/public/traces_public.jsonl    episode_id + mode + observable_trace + trace_text   (NO labels)
  data/private/traces_hidden.jsonl   episode_id (the held-out eval subset)
  data/private/measured_outcomes.jsonl  episode_id + reward_by_action + correct_action    (ground truth)

Public = what the target agent (gpt-oss selector) sees. Private = ground truth for evaluate.py.
No fabrication: every reward comes from the real-rerun kernel cache.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "gpt_oss"))
from lever_io import load_cache, cost_adjusted_best  # noqa: E402

DEFAULT_CACHE = os.path.join(PROJ, "gpt_oss", "data", "out", "kernel_cache.jsonl")
EVAL_SEEDS = 3  # held-out eval subset (highest seed indices) the agent is scored on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="measured kernel cache to build from")
    ap.add_argument("--eval-seeds", type=int, default=EVAL_SEEDS)
    args = ap.parse_args()
    cache = load_cache(args.cache)
    seeds = sorted({r["seed"] for r in cache})
    eval_set = set(seeds[-args.eval_seeds:]) if args.eval_seeds else set(seeds)

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
                "correct_action": r.get("correct_action") or cost_adjusted_best(r["reward_by_action"]),
            }) + "\n")

    with open(os.path.join(pub, "sample_submission.json"), "w") as f:
        json.dump({"predictions": [
            {"episode_id": cache[0]["episode_id"], "action": "H_THEN_W",
             "reason": "example: large weak-minus-heldout pass gap = cheat kernel passed a weak "
                       "verifier -> strengthen verifier then re-select."}
        ]}, f, indent=2)

    print(f"[trimul task] public traces: {len(cache)}; hidden eval episodes: {n_hidden} "
          f"(seeds {sorted(eval_set)}) from {args.cache}")


if __name__ == "__main__":
    main()
