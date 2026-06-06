"""Build a larger, CLASS-BALANCED HARD SFT training set (hard_balanced_v2) without touching the
fixed held-out eval set.

Why: the 20-epoch HARD LoRA collapsed to constant-H because the train labels are H-heavy and tiny
(48 ex). A constant-H policy then reaches the eval majority rate. Fix = more episodes + balanced
labels so the loss cannot be satisfied by a constant.

Protocol (no eval-set change, no leakage):
  - EVAL stays the original `gpt_oss/data/out/hard/trace_action_eval.jsonl` (hard_cache seed 2, 24 ep).
  - TRAIN is drawn from a larger cache (e.g. hard_cache_v2.jsonl, reps=12) EXCLUDING the eval fold
    (seed == --eval-seed, default 2) AND asserting zero episode_id overlap with the eval file.
  - Balance by downsampling each over-represented label to the minority count (deterministic:
    sort by episode_id, take first K — no RNG).

Outputs:
  gpt_oss/data/out/hard_v2/trace_action_train.jsonl   (balanced SFT rows, {"messages":...})
  gpt_oss/data/out/hard_v2/manifest.json              (provenance: gen cmd, seeds, splits, dists)

Usage:
  python gpt_oss/data/build_balanced_v2.py \
     --cache gpt_oss/data/out/hard_cache_v2.jsonl \
     --eval-file gpt_oss/data/out/hard/trace_action_eval.jsonl --eval-seed 2 --reps 12
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lever_io import load_cache, cost_adjusted_best, PROJ  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sft_dataset import to_sft_row  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(PROJ, "gpt_oss/data/out/hard_cache_v2.jsonl"))
    ap.add_argument("--eval-file", default=os.path.join(PROJ, "gpt_oss/data/out/hard/trace_action_eval.jsonl"))
    ap.add_argument("--eval-seed", type=int, default=2, help="seed fold held out for eval (excluded from train)")
    ap.add_argument("--reps", type=int, default=12, help="recorded in manifest (generation reps)")
    ap.add_argument("--steps", type=int, default=300, help="recorded in manifest (generation steps)")
    ap.add_argument("--out", default=os.path.join(PROJ, "gpt_oss/data/out/hard_v2"))
    ap.add_argument("--cap-per-class", type=int, default=None, help="optional max per class (else = minority count)")
    args = ap.parse_args()

    rows = load_cache(args.cache)
    # eval episode ids (the fixed benchmark) — exclude from train + leakage assert
    eval_ids = set()
    with open(args.eval_file) as f:
        for line in f:
            if line.strip():
                eval_ids.add(json.loads(line)["episode_id"])

    # candidate train rows: exclude eval fold + any id that appears in the eval file
    cand = [r for r in rows if r["seed"] != args.eval_seed and r["episode_id"] not in eval_ids]
    by_label = collections.defaultdict(list)
    for r in cand:
        by_label[cost_adjusted_best(r["reward_by_action"])].append(r)
    for lab in by_label:
        by_label[lab].sort(key=lambda r: r["episode_id"])   # deterministic order, no RNG

    counts = {k: len(v) for k, v in by_label.items()}
    k_per = args.cap_per_class or (min(counts.values()) if counts else 0)
    balanced = []
    for lab, items in by_label.items():
        balanced.extend(items[:k_per])
    balanced.sort(key=lambda r: r["episode_id"])

    os.makedirs(args.out, exist_ok=True)
    train_path = os.path.join(args.out, "trace_action_train.jsonl")
    with open(train_path, "w") as f:
        for r in balanced:
            f.write(json.dumps(to_sft_row(r)) + "\n")

    bal_label = collections.Counter(cost_adjusted_best(r["reward_by_action"]) for r in balanced)
    bal_mode = collections.Counter(r["mode"] for r in balanced)
    manifest = {
        "generation_command": f"python experiments/hard_task.py --reps {args.reps} --steps {args.steps} --out {os.path.relpath(args.cache, PROJ)}",
        "source_cache": os.path.relpath(args.cache, PROJ),
        "eval_file_FIXED": os.path.relpath(args.eval_file, PROJ),
        "eval_seed_excluded": args.eval_seed,
        "train_seeds": sorted({r["seed"] for r in balanced}),
        "raw_candidate_label_counts": counts,
        "per_class_kept": k_per,
        "balanced_label_distribution": dict(bal_label),
        "balanced_mode_distribution": dict(bal_mode),
        "n_train": len(balanced),
        "n_eval_fixed": len(eval_ids),
        "leakage_check": "train excludes eval-seed fold AND eval episode_ids; validate with validate_dataset.py",
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"raw candidate label counts: {counts}")
    print(f"kept {k_per}/class -> balanced {dict(bal_label)}  (n_train={len(balanced)})")
    print(f"train seeds: {manifest['train_seeds']}  | eval fold {args.eval_seed} excluded ({len(eval_ids)} eval ids)")
    print(f"wrote {train_path}\nwrote {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
