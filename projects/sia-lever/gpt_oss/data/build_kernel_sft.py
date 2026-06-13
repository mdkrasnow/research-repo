"""Build a CLASS-BALANCED kernel (TriMul GPU) SFT train set + a held-out eval fold.

Mirrors build_balanced_v2.py, but the source is the GPU-kernel cache (REAL CUDA latency,
gpt_oss/data/out/kernel_cache_gpu.jsonl). Schema is identical to hard_cache.

Gold dist is imbalanced + 2 classes (H_THEN_W 240 / H 80 across 10 seeds). A constant-H_THEN_W
policy already reaches the eval majority — so TRAIN must be class-balanced (downsample the majority
to the minority count, deterministic sort-by-episode-id, NO rng), exactly like the rung-2 fix
(documentation/ladder_findings.md). One seed fold is held out for eval; zero episode_id leakage
is asserted.

Outputs (gpt_oss/data/out/kernel_v1/):
  trace_action_train.jsonl   balanced SFT rows {"messages":..., bookkeeping...}
  trace_action_eval.jsonl    held-out eval fold (NOT balanced; full eval distribution)
  manifest.json              provenance: source, seeds, splits, dists, leakage check

Usage:
  python gpt_oss/data/build_kernel_sft.py --eval-seed 9
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
    ap.add_argument("--cache", default=os.path.join(PROJ, "gpt_oss/data/out/kernel_cache_gpu.jsonl"))
    ap.add_argument("--eval-seed", type=int, default=9, help="seed fold held out for eval (excluded from train)")
    ap.add_argument("--out", default=os.path.join(PROJ, "gpt_oss/data/out/kernel_v1"))
    ap.add_argument("--cap-per-class", type=int, default=None, help="optional max per class (else = minority count)")
    args = ap.parse_args()

    rows = load_cache(args.cache)

    # eval fold = one held-out seed (full, unbalanced — reflects true deploy distribution)
    eval_rows = [r for r in rows if r["seed"] == args.eval_seed]
    eval_ids = {r["episode_id"] for r in eval_rows}

    # candidate train rows: everything NOT in the eval fold
    cand = [r for r in rows if r["seed"] != args.eval_seed and r["episode_id"] not in eval_ids]
    by_label = collections.defaultdict(list)
    for r in cand:
        by_label[cost_adjusted_best(r["reward_by_action"])].append(r)
    for lab in by_label:
        by_label[lab].sort(key=lambda r: r["episode_id"])  # deterministic order, no RNG

    counts = {k: len(v) for k, v in by_label.items()}
    k_per = args.cap_per_class or (min(counts.values()) if counts else 0)
    balanced = []
    for lab, items in by_label.items():
        balanced.extend(items[:k_per])
    balanced.sort(key=lambda r: r["episode_id"])

    os.makedirs(args.out, exist_ok=True)
    train_path = os.path.join(args.out, "trace_action_train.jsonl")
    eval_path = os.path.join(args.out, "trace_action_eval.jsonl")
    with open(train_path, "w") as f:
        for r in balanced:
            f.write(json.dumps(to_sft_row(r)) + "\n")
    eval_sorted = sorted(eval_rows, key=lambda r: r["episode_id"])
    with open(eval_path, "w") as f:
        for r in eval_sorted:
            f.write(json.dumps(to_sft_row(r)) + "\n")

    # leakage assert (belt-and-suspenders; validate_dataset.py also checks)
    tr_ids = {r["episode_id"] for r in balanced}
    overlap = tr_ids & eval_ids
    assert not overlap, f"LEAKAGE: {len(overlap)} ids in both train and eval"

    bal_label = collections.Counter(cost_adjusted_best(r["reward_by_action"]) for r in balanced)
    bal_mode = collections.Counter(r["mode"] for r in balanced)
    ev_label = collections.Counter(cost_adjusted_best(r["reward_by_action"]) for r in eval_rows)
    manifest = {
        "source_cache": os.path.relpath(args.cache, PROJ),
        "source_note": "REAL CUDA latency GPU-kernel (TriMul) cache",
        "eval_seed_held_out": args.eval_seed,
        "train_seeds": sorted({r["seed"] for r in balanced}),
        "raw_candidate_label_counts": counts,
        "per_class_kept": k_per,
        "balanced_label_distribution": dict(bal_label),
        "balanced_mode_distribution": dict(bal_mode),
        "eval_label_distribution": dict(ev_label),
        "n_train": len(balanced),
        "n_eval": len(eval_rows),
        "leakage_check": "train excludes eval-seed fold; zero episode_id overlap asserted",
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"raw candidate label counts: {counts}")
    print(f"kept {k_per}/class -> balanced train {dict(bal_label)}  (n_train={len(balanced)})")
    print(f"eval fold seed {args.eval_seed}: {dict(ev_label)}  (n_eval={len(eval_rows)})")
    print(f"train seeds: {manifest['train_seeds']}")
    print(f"leakage overlap: {len(overlap)} (0 expected)")
    print(f"wrote {train_path}\nwrote {eval_path}\nwrote {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
