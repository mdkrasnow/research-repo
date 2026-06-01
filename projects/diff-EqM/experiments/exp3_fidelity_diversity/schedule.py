"""Deterministic shared label + seed schedule for Experiment 3.

The whole experiment hinges on vanilla and ANM seeing the IDENTICAL inputs:
the same requested class label and the same initial-noise seed at every sample
index i. This module builds that schedule once, saves it to JSON, and both arms
reload it. The schedule is independent of GPU count: sample index i always maps
to (label_schedule[i], base_seed + i) regardless of how generation is sharded.

Confound controlled: sample_gd.py draws labels with torch.randint per batch
(not balanced, not deterministic, not shared). Exp3 replaces that with this
fixed balanced schedule.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def make_balanced_label_schedule(num_classes, samples_per_class, shuffle_seed):
    """Balanced label vector: each class appears exactly samples_per_class times,
    then deterministically shuffled. Length = num_classes * samples_per_class."""
    labels = np.repeat(np.arange(num_classes, dtype=np.int64), samples_per_class)
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(labels)
    return labels


def seed_schedule(base_seed, n):
    """Per-sample noise seeds: index i -> base_seed + i."""
    return (base_seed + np.arange(n, dtype=np.int64))


def build_schedule(num_classes, samples_per_class, base_seed, shuffle_seed):
    labels = make_balanced_label_schedule(num_classes, samples_per_class, shuffle_seed)
    seeds = seed_schedule(base_seed, len(labels))
    return {
        "num_classes": int(num_classes),
        "samples_per_class": int(samples_per_class),
        "num_samples": int(len(labels)),
        "base_seed": int(base_seed),
        "shuffle_seed": int(shuffle_seed),
        "labels": labels.tolist(),
        "seeds": seeds.tolist(),
    }


def save_schedule(schedule, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schedule))
    return path


def load_schedule(path):
    schedule = json.loads(Path(path).read_text())
    schedule["labels"] = np.asarray(schedule["labels"], dtype=np.int64)
    schedule["seeds"] = np.asarray(schedule["seeds"], dtype=np.int64)
    return schedule


def schedule_hash(schedule):
    """Stable hash over (labels, seeds) so vanilla/ANM equality is verifiable.

    A mismatch between the two arms' schedule hashes is a hard failure: the
    paired-difference estimand is only valid if the inputs are identical.
    """
    labels = np.asarray(schedule["labels"], dtype=np.int64).tobytes()
    seeds = np.asarray(schedule["seeds"], dtype=np.int64).tobytes()
    h = hashlib.sha256()
    h.update(labels)
    h.update(seeds)
    return h.hexdigest()[:16]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build and save the Exp3 shared schedule")
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--samples-per-class", type=int, default=50)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sched = build_schedule(args.num_classes, args.samples_per_class,
                           args.base_seed, args.shuffle_seed)
    save_schedule(sched, args.out)
    print(f"schedule: N={sched['num_samples']} classes={sched['num_classes']} "
          f"per_class={sched['samples_per_class']} hash={schedule_hash(sched)}")
    print(f"saved -> {args.out}")
