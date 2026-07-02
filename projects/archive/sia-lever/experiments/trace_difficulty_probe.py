#!/usr/bin/env python3
"""Trace-difficulty probe — the cheap pre-flight gate BEFORE the GPU (Phase 4) rung.

Question it answers: is the lever-attribution task actually learnable, or is the answer just
printed in the trace? If a tiny threshold model already solves it on held-out episodes, a
gpt-oss-120b LoRA cannot demonstrate "learned attribution" and the H200 spend is wasted.

Two feature sets, both scored with a proper TRAIN/EVAL split by seed (train 0..6, eval 7..9):

  WITH_BOOLEANS  the current trace: the derived giveaway flags
                 (shortcut_cheat_signature, harness_accepts_known_good_model, predicts_clean, ...)
  RAW_ONLY       hardened trace: raw numbers only, NO derived booleans, NO prose hints
                 (clean_mse, neg_control_mse, composition_error, reference_model_score_under_harness)

For each set we fit the SMALLEST possible learners (depth-1 stump, depth-2 pair-rule) on TRAIN and
score on EVAL. "Headroom for a big model" only exists if a tiny model CANNOT already solve it.

CPU, < 1 s, no deps beyond stdlib. Run: python experiments/trace_difficulty_probe.py
"""

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "gpt_oss"))
from lever_io import cost_adjusted_best  # noqa: E402

CACHE = os.path.join(PROJ, "gpt_oss", "data", "out", "action_outcome_cache.jsonl")

RAW_ONLY = ["clean_mse", "neg_control_mse", "composition_error",
            "reference_model_score_under_harness"]
WITH_BOOLEANS = RAW_ONLY + ["predicts_clean", "solves_broken_symmetry",
                            "shortcut_cheat_signature", "harness_accepts_known_good_model"]


def _bins(vals):
    """Candidate split points: midpoints between sorted unique values (+ the values for bools)."""
    u = sorted(set(vals))
    if len(u) <= 3:
        return u
    return [(u[i] + u[i + 1]) / 2 for i in range(len(u) - 1)]


def _bin_id(v, t1, t2):
    return 0 if v <= t1 else (1 if v <= t2 else 2)


def fit_stump(rows, feat, labels):
    """Best single feature, 2 thresholds -> 3 bins -> majority label per bin. Fit on rows."""
    vals = [r["observable_trace"][feat] for r in rows]
    cands = _bins(vals)
    best = (-1.0, None)
    pairs = list(itertools.combinations(cands, 2)) or [(cands[0], cands[0])] if cands else []
    if len(cands) < 2:
        pairs = [(cands[0] if cands else 0.0, cands[0] if cands else 0.0)]
    for t1, t2 in pairs:
        groups = defaultdict(list)
        for v, l in zip(vals, labels):
            groups[_bin_id(v, t1, t2)].append(l)
        maj = {k: max(set(g), key=g.count) for k, g in groups.items()}
        acc = sum(maj[_bin_id(v, t1, t2)] == l for v, l in zip(vals, labels)) / len(labels)
        if acc > best[0]:
            best = (acc, ("stump", feat, t1, t2, maj))
    return best


def fit_pair(rows, feats, labels):
    """Best 2-feature rule: bin each feature into <=t / >t, 4 cells, majority per cell. Fit on rows."""
    best = (-1.0, None)
    for fa, fb in itertools.combinations(feats, 2):
        va = [r["observable_trace"][fa] for r in rows]
        vb = [r["observable_trace"][fb] for r in rows]
        for ta in _bins(va):
            for tb in _bins(vb):
                groups = defaultdict(list)
                for a, b, l in zip(va, vb, labels):
                    groups[(a <= ta, b <= tb)].append(l)
                maj = {k: max(set(g), key=g.count) for k, g in groups.items()}
                acc = sum(maj[(a <= ta, b <= tb)] == l
                          for a, b, l in zip(va, vb, labels)) / len(labels)
                if acc > best[0]:
                    best = (acc, ("pair", fa, fb, ta, tb, maj))
    return best


def score(model, rows, labels):
    kind = model[0]
    if kind == "stump":
        _, feat, t1, t2, maj = model
        d = {k: maj.get(k) for k in (0, 1, 2)}
        ok = 0
        for r, l in zip(rows, labels):
            b = _bin_id(r["observable_trace"][feat], t1, t2)
            ok += int(d.get(b) == l)
        return ok / len(labels)
    _, fa, fb, ta, tb, maj = model
    ok = 0
    for r, l in zip(rows, labels):
        cell = (r["observable_trace"][fa] <= ta, r["observable_trace"][fb] <= tb)
        ok += int(maj.get(cell) == l)
    return ok / len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE, help="action_outcome_cache jsonl to probe")
    ap.add_argument("--eval-seeds", type=int, default=3, help="highest-N seeds held out for eval")
    args = ap.parse_args()

    cache = [json.loads(l) for l in open(args.cache) if l.strip()]
    # only keep raw features present in this cache (hard cache may add/rename)
    present = set(cache[0]["observable_trace"].keys())
    global RAW_ONLY, WITH_BOOLEANS
    RAW_ONLY = [f for f in RAW_ONLY if f in present]
    WITH_BOOLEANS = [f for f in WITH_BOOLEANS if f in present]
    seeds = sorted({r["seed"] for r in cache})
    eval_seeds = set(seeds[-args.eval_seeds:])
    train = [r for r in cache if r["seed"] not in eval_seeds]
    ev = [r for r in cache if r["seed"] in eval_seeds]
    gold = {r["episode_id"]: cost_adjusted_best(r["reward_by_action"]) for r in cache}
    ytr = [gold[r["episode_id"]] for r in train]
    yev = [gold[r["episode_id"]] for r in ev]

    print("=" * 70)
    print(f"Trace-difficulty probe  (train {len(train)} ep, eval {len(ev)} ep held-out, 3 classes)")
    print(f"random-guess baseline = {1/3:.2f}")
    print("=" * 70)

    out = {}
    for name, feats in [("WITH_BOOLEANS (current trace)", WITH_BOOLEANS),
                        ("RAW_ONLY (hardened: numbers only)", RAW_ONLY)]:
        # best stump and best pair, both CHOSEN ON TRAIN, SCORED ON EVAL
        s_tr, s_model = max((fit_stump(train, f, ytr) for f in feats), key=lambda x: x[0])
        s_ev = score(s_model, ev, yev)
        p_tr, p_model = fit_pair(train, feats, ytr)
        p_ev = score(p_model, ev, yev)
        out[name] = (s_tr, s_ev, p_tr, p_ev)
        print(f"\n{name}")
        print(f"  depth-1 stump (1 feature) : train {s_tr:.2f}   eval {s_ev:.2f}   [{s_model[1]}]")
        print(f"  depth-2 pair  (2 features): train {p_tr:.2f}   eval {p_ev:.2f}   "
              f"[{p_model[1]} & {p_model[2]}]")

    # --- reference points that bound the headroom ---
    yall = ytr + yev
    majority = max(set(yall), key=yall.count)
    maj_floor = sum(1 for y in yev if y == majority) / len(yev)   # always-majority on eval
    # config-oracle ceiling: best gold achievable from the LATENT config (if recorded)
    cfg_ceiling = None
    if "config" in cache[0]:
        groups = defaultdict(list)
        for r in cache:
            groups[tuple(sorted(r["config"].items()))].append(gold[r["episode_id"]])
        cmaj = {k: max(set(v), key=v.count) for k, v in groups.items()}
        cfg_ceiling = sum(cmaj[tuple(sorted(r["config"].items()))] == gold[r["episode_id"]]
                          for r in cache) / len(cache)

    print("\n" + "=" * 70)
    ro = out["RAW_ONLY (hardened: numbers only)"]
    tiny = ro[3]
    print("HEADROOM (the band a capable model could learn)")
    print(f"  always-majority floor (eval)   : {maj_floor:.2f}")
    print(f"  tiny 2-feature model (eval)    : {tiny:.2f}")
    if cfg_ceiling is not None:
        print(f"  latent-config ceiling          : {cfg_ceiling:.2f}  "
              f"(rest = irreducible seed noise)")
    print("VERDICT")
    if tiny >= 0.99:
        print("  => Tiny model already solves it on held-out. NO headroom -> DO NOT spend the H200.")
    elif cfg_ceiling is not None and (cfg_ceiling - max(tiny, maj_floor)) >= 0.15:
        print(f"  => Real headroom: tiny/majority ~{max(tiny, maj_floor):.2f} vs achievable "
              f"~{cfg_ceiling:.2f}. A capable selector has room a threshold rule can't reach. "
              "Phase 4 is INFORMATIVE.")
    elif tiny <= max(maj_floor, 0.5) + 0.05:
        print("  => Tiny model stuck near the majority floor, but ceiling unclear. Likely headroom; "
              "confirm an oracle/rule beats the floor before GPU.")
    else:
        print("  => Partial gap. Decide if the remaining gap justifies GPU.")
    print("=" * 70)


if __name__ == "__main__":
    main()
