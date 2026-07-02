"""Build GRPO prompts from the measured cache.

GRPO needs prompts + a reward function over sampled completions. We emit one prompt per TRAIN
episode plus the per-action measured reward table so the reward fn can score a sampled action by
its REAL measured outcome (with cost adjustment). No fabricated rewards.

Output: grpo_prompts_train.jsonl. Each row:
  {prompt_messages, episode_id, mode, seed, reward_by_action (incl NOOP/KILL), w_cost}
The trainer (train_lora_grpo.py) parses the sampled action and looks up reward_by_action.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lever_io import build_messages, load_cache, PROJ  # noqa: E402

# action token (as emitted by the model) -> measured-cache key
TOKEN_TO_KEY = {"H": "H", "W": "W", "H_THEN_W": "H_THEN_W", "PROMOTE": "NOOP", "KILL": "KILL"}
W_RETRAINS = {"H": 0, "W": 1, "H_THEN_W": 1, "PROMOTE": 0, "KILL": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--w-cost", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(PROJ, "gpt_oss", "data", "out"))
    args = ap.parse_args()

    rows = load_cache(args.cache)
    seeds = sorted({r["seed"] for r in rows})
    eval_set = set(seeds[-args.eval_seeds:]) if args.eval_seeds else set()

    out = []
    for r in rows:
        if r["seed"] in eval_set:
            continue
        out.append({
            "episode_id": r["episode_id"], "mode": r["mode"], "seed": r["seed"],
            "prompt_messages": build_messages(r["trace_text"]),
            "reward_by_action": r["reward_by_action"],
            "token_to_key": TOKEN_TO_KEY,
            "w_retrains": W_RETRAINS,
            "w_cost": args.w_cost,
        })

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "grpo_prompts_train.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"GRPO: {len(out)} train prompts (eval seeds {sorted(eval_set)} held out)")


if __name__ == "__main__":
    main()
