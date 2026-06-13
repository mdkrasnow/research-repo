#!/usr/bin/env python3
"""SIA custom-task evaluator for SIA-Lever.

Contract (per SIA EVALUATION_GUIDE):
  - exposes evaluate(submission_path: Path) -> dict
  - the orchestrator finds the submission in --gen-dir, calls evaluate, writes results.json
Standalone:
  python evaluate.py --gen-dir <dir with submission.json>
  python evaluate.py --submission <submission.json>

Scoring uses MEASURED outcomes only (data/private/measured_outcomes.jsonl, from real reruns).
If an episode's outcome is missing it FAILS LOUDLY. No hardcoded transition effects.
"""

import argparse
import json
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent.parent          # sia_task/
PRIV = TASK_DIR / "data" / "private"

ACTIVE = ["H", "W", "H_THEN_W"]
W_RETRAINS = {"H": 0, "W": 1, "H_THEN_W": 1, "PROMOTE": 0, "KILL": 0}
W_COST = 0.05
VALID_ACTIONS = ["H", "W", "H_THEN_W", "PROMOTE", "KILL"]


def _cost_adjusted_best(rba):
    adj = {k: rba[k] - W_COST * W_RETRAINS[k] for k in ACTIVE}
    m = max(adj.values())
    for k in ACTIVE:
        if adj[k] >= m - 1e-9:
            return k
    return ACTIVE[0]


def _outcome(action, rba):
    if action in ACTIVE:
        return rba[action] - W_COST * W_RETRAINS[action]
    return 0.0   # PROMOTE/KILL: no real intervention on a genuine failure (documented floor)


def _load_outcomes():
    out = {}
    with open(PRIV / "measured_outcomes.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["episode_id"]] = r
    if not out:
        raise RuntimeError("measured_outcomes.jsonl empty — run sia_task/build_task_data.py")
    return out


def _find_submission(gen_dir: Path):
    for cand in [gen_dir / "submission.json", gen_dir / "results" / "submission.json"]:
        if cand.is_file():
            return cand
    for pat in ("submission*.json", "results/*.json", "*.json"):
        m = sorted(gen_dir.glob(pat))
        if m:
            return m[0]
    return None


def _worst_case_results(outcomes, error):
    """Fail-soft: a malformed/unreadable submission is scored as worst-case (every episode
    KILL -> 0.0 outcome) rather than crashing the SIA orchestrator. The error is surfaced."""
    regrets = []
    for oc in outcomes.values():
        rba = oc["reward_by_action"]
        best_score = _outcome(_cost_adjusted_best(rba), rba)
        regrets.append(best_score - _outcome("KILL", rba))
    n = len(outcomes)
    return {
        "n_examples": n,
        "lever_accuracy": 0.0,
        "mean_regret": sum(regrets) / len(regrets) if regrets else 0.0,
        "max_regret": max(regrets) if regrets else 0.0,
        "hidden_structural_score": 1.0 - (sum(regrets) / len(regrets) if regrets else 0.0),
        "invalid_json_rate": 1.0,
        "missing_rate": 1.0,
        "action_distribution": {a: 0 for a in VALID_ACTIONS} | {"MISSING/INVALID": n},
        "per_mode_accuracy": {},
        "error": str(error),
    }


def evaluate(submission_path: Path) -> dict:
    submission_path = Path(submission_path)
    outcomes = _load_outcomes()
    # Fail-soft: a target agent that writes malformed/unreadable JSON must not crash the
    # evaluator (the SIA orchestrator contract expects a results dict). Score worst-case instead.
    try:
        sub = json.loads(submission_path.read_text())
        if not isinstance(sub, dict):
            raise ValueError("submission is not a JSON object")
    except (json.JSONDecodeError, ValueError, OSError) as e:
        return _worst_case_results(outcomes, f"unreadable submission: {e}")

    # Detect duplicate episode_ids (silent dedupe would otherwise hide conflicting predictions).
    raw_preds = sub.get("predictions", []) or []
    seen_ids, dup_ids = set(), set()
    preds = {}
    for p in raw_preds:
        if not isinstance(p, dict):
            continue
        eid = p.get("episode_id")
        if eid in seen_ids:
            dup_ids.add(eid)
        seen_ids.add(eid)
        preds.setdefault(eid, p)   # keep FIRST occurrence (deterministic), flag the dup

    n = len(outcomes)
    correct, invalid, missing = 0, 0, 0
    regrets = []
    per_mode = {}
    dist = {a: 0 for a in VALID_ACTIONS}
    dist["MISSING/INVALID"] = 0
    for eid, oc in outcomes.items():
        rba = oc["reward_by_action"]
        for a in ACTIVE:
            if a not in rba:
                raise RuntimeError(f"missing measured outcome for {a} on {eid} — cannot score")
        gold = oc.get("correct_action") or _cost_adjusted_best(rba)
        p = preds.get(eid)
        act = (p or {}).get("action")
        if act not in VALID_ACTIONS:
            if p is None:
                missing += 1
            else:
                invalid += 1
            dist["MISSING/INVALID"] += 1
            act = "KILL"   # no usable action -> worst-case
        else:
            dist[act] += 1
        best_score = _outcome(_cost_adjusted_best(rba), rba)
        regrets.append(best_score - _outcome(act, rba))
        correct += int(act == gold)
        pm = per_mode.setdefault(oc["mode"], {"correct": 0, "n": 0})
        pm["n"] += 1
        pm["correct"] += int(act == gold)

    results = {
        "n_examples": n,
        "lever_accuracy": correct / n if n else 0.0,
        "mean_regret": sum(regrets) / len(regrets) if regrets else 0.0,
        "max_regret": max(regrets) if regrets else 0.0,
        "hidden_structural_score": 1.0 - (sum(regrets) / len(regrets) if regrets else 0.0),
        "invalid_json_rate": invalid / n if n else 0.0,
        "missing_rate": missing / n if n else 0.0,
        "action_distribution": dist,
        "per_mode_accuracy": {m: v["correct"] / v["n"] for m, v in per_mode.items()},
        "duplicate_episode_ids": sorted(dup_ids),
    }
    # goodhart_index: only meaningful if a public-vs-hidden gap is available; here we expose the
    # gap between naive "always predict-clean" (PROMOTE) regret and the chosen regret is not defined
    # per-episode, so we omit it rather than fabricate.
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=None)
    ap.add_argument("--submission", default=None)
    args = ap.parse_args()

    if args.submission:
        sub = Path(args.submission)
    elif args.gen_dir:
        sub = _find_submission(Path(args.gen_dir))
        if sub is None:
            raise FileNotFoundError(f"no submission found in {args.gen_dir}")
    else:
        raise SystemExit("pass --gen-dir or --submission")

    results = evaluate(sub)
    out_dir = Path(args.gen_dir) if args.gen_dir else sub.parent
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
