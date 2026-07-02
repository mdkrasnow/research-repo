#!/usr/bin/env python3
"""Learner-ceiling probe: can a TINY trained model (not a threshold stump) solve the lever task
from the trace? This is the control the difficulty stump can't give (stumps overfit / underfit).

If a 2-layer MLP on the 4 trace numbers already hits the ~0.81 latent ceiling on held-out, then:
  - the headroom IS capturable by a tiny CPU model -> a 120B LoRA is overkill, and
  - base gpt-oss-120b (a far stronger learner) will likely already be near-ceiling zero-shot
    -> a base-vs-LoRA GPU experiment risks a NULL result even on the hard task.

Trains on seed folds [0..n-2], evaluates on the last fold. CPU, seconds. torch only.

Run: python experiments/learner_ceiling_probe.py --cache gpt_oss/data/out/hard_cache.jsonl
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "gpt_oss"))
from lever_io import cost_adjusted_best  # noqa: E402

RAW = ["clean_mse", "neg_control_mse", "composition_error", "reference_model_score_under_harness"]
BOOLS = ["predicts_clean", "solves_broken_symmetry", "shortcut_cheat_signature",
         "harness_accepts_known_good_model"]
ACTIONS = ["H", "W", "H_THEN_W"]


def featurize(rows, feats):
    X, y = [], []
    for r in rows:
        ot = r["observable_trace"]
        X.append([float(ot[f]) for f in feats])
        y.append(ACTIONS.index(cost_adjusted_best(r["reward_by_action"])))
    return torch.tensor(X), torch.tensor(y)


def standardize(Xtr, Xev):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    return (Xtr - mu) / sd, (Xev - mu) / sd


def train_eval(Xtr, ytr, Xev, yev, seed=0, epochs=400, wd=1e-3):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 16), nn.ReLU(), nn.Linear(16, len(ACTIONS)))
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); lossf(net(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        tr = (net(Xtr).argmax(1) == ytr).float().mean().item()
        ev = (net(Xev).argmax(1) == yev).float().mean().item()
    return tr, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(PROJ, "gpt_oss", "data", "out", "hard_cache.jsonl"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.cache) if l.strip()]
    seeds = sorted({r["seed"] for r in rows})
    ev_seed = seeds[-1]
    tr_rows = [r for r in rows if r["seed"] != ev_seed]
    ev_rows = [r for r in rows if r["seed"] == ev_seed]

    yall = [cost_adjusted_best(r["reward_by_action"]) for r in rows]
    maj = max(set(yall), key=yall.count)
    maj_floor = sum(1 for r in ev_rows
                    if cost_adjusted_best(r["reward_by_action"]) == maj) / len(ev_rows)

    print("=" * 66)
    print(f"Learner-ceiling probe  ({len(tr_rows)} train / {len(ev_rows)} eval, held-out seed {ev_seed})")
    print(f"majority-floor (eval) = {maj_floor:.2f}   random = 0.33")
    print("=" * 66)
    for name, feats in [("raw 4 numbers", RAW), ("raw + booleans (8)", RAW + BOOLS)]:
        Xtr, ytr = featurize(tr_rows, feats)
        Xev, yev = featurize(ev_rows, feats)
        Xtr, Xev = standardize(Xtr, Xev)
        # median over a few inits (tiny data -> variance)
        evs = []
        for s in range(5):
            tr, ev = train_eval(Xtr, ytr, Xev, yev, seed=s)
            evs.append(ev)
        evs.sort()
        print(f"  {name:20s}: eval acc (median of 5) = {evs[2]:.2f}   range [{evs[0]:.2f},{evs[-1]:.2f}]")
    print("=" * 66)
    print("Read: if eval ~0.8 -> a TINY MLP captures the headroom; 120B LoRA is overkill and base")
    print("      gpt-oss-120b is likely already near-ceiling zero-shot (base-vs-LoRA may be NULL).")


if __name__ == "__main__":
    main()
