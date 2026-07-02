#!/usr/bin/env python3
"""Robustness / integrity regression for SIA-Lever (adversarial-audit additions).

Cheap, CPU-only, no GPU/endpoint. Run before every commit alongside run_cpu_regression.sh.
Covers the audit's robustness checklist:

  1. evaluator robustness   valid / missing / invalid-action / duplicate / malformed-JSON / empty
  2. no-private-data-access only the evaluator + task builder may read sia_task/data/private/**
  3. cache validation       schema, reward bounds, all levers present, meta hash matches bytes
  4. train/eval split        SFT + DPO eval seeds are disjoint from train seeds (by seed)
  5. gold-definition check   surfaces episodes where pre-registered correct != cost-adjusted best
  6. leak/seed/W_COST sweeps pointers (full sweeps live in phase3.py + leak_sweep.py)

Exit code 0 = all PASS. Non-zero = at least one FAIL (prints which).
"""

import glob
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "gpt_oss"))
from lever_io import cost_adjusted_best  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def _load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "sl_evaluate", os.path.join(PROJ, "sia_task", "data", "public", "evaluate.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


# ---------------------------------------------------------------- 1. evaluator robustness
def test_evaluator_robustness():
    E = _load_evaluator()
    priv = {r["episode_id"]: r for r in _load_jsonl(
        os.path.join(PROJ, "sia_task", "data", "private", "measured_outcomes.jsonl"))}
    golds = [{"episode_id": k, "action": v["correct_action"]} for k, v in priv.items()]

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        def run(obj=None, raw=None):
            p = d / "s.json"
            p.write_text(raw if raw is not None else json.dumps(obj))
            return E.evaluate(p)

        r = run({"predictions": golds})
        check("evaluator/valid_all_gold", r["lever_accuracy"] == 1.0 and r["mean_regret"] == 0.0,
              f"acc={r['lever_accuracy']:.2f} regret={r['mean_regret']:.3f}")

        r = run({"predictions": []})
        check("evaluator/empty_is_worst", r["missing_rate"] == 1.0 and r["lever_accuracy"] == 0.0)

        r = run({"predictions": [{"episode_id": k, "action": "BOGUS"} for k in priv]})
        check("evaluator/invalid_action_counted", r["invalid_json_rate"] == 1.0)

        r = run({"predictions": golds + golds})
        check("evaluator/duplicate_flagged",
              len(r.get("duplicate_episode_ids", [])) == len(priv) and r["lever_accuracy"] == 1.0,
              f"dups={len(r.get('duplicate_episode_ids', []))}")

        # malformed JSON must FAIL-SOFT (return a dict with an error), not raise
        try:
            r = run(raw='{"predictions": [ {"episode_id":')
            check("evaluator/malformed_failsoft",
                  isinstance(r, dict) and "error" in r and r["invalid_json_rate"] == 1.0,
                  "returned worst-case dict")
        except Exception as ex:
            check("evaluator/malformed_failsoft", False, f"raised {type(ex).__name__}: {ex}")

        r = run({"predictions": golds[:3]})
        check("evaluator/partial_missing", 0.0 < r["missing_rate"] < 1.0)


# ---------------------------------------------------------------- 2. no private data access
def test_no_private_access():
    allowed = {
        os.path.join(PROJ, "sia_task", "data", "public", "evaluate.py"),
        os.path.join(PROJ, "sia_task", "build_task_data.py"),
        os.path.abspath(__file__),   # this harness references the path only to TEST access
    }
    offenders = []
    for py in glob.glob(os.path.join(PROJ, "**", "*.py"), recursive=True):
        if "__pycache__" in py or "vendor" in py or py in allowed:
            continue
        try:
            txt = open(py).read()
        except Exception:
            continue
        if "data/private" in txt or "measured_outcomes.jsonl" in txt or "traces_hidden.jsonl" in txt:
            offenders.append(os.path.relpath(py, PROJ))
    check("no_private_access/only_evaluator_and_builder", not offenders,
          f"offenders={offenders}" if offenders else "only evaluator + builder reference private/")

    # the reference target agent must read ONLY public traces
    ref = os.path.join(PROJ, "sia_task", "reference", "reference_target_agent.py")
    if os.path.exists(ref):
        txt = open(ref).read()
        check("no_private_access/reference_agent_public_only",
              "private" not in txt and "correct_action" not in txt and "measured_outcomes" not in txt)


# ---------------------------------------------------------------- 3. cache validation
def test_cache_validation():
    cache_path = os.path.join(PROJ, "gpt_oss", "data", "out", "action_outcome_cache.jsonl")
    rows = _load_jsonl(cache_path)
    check("cache/nonempty", len(rows) > 0, f"{len(rows)} rows")

    ok_schema = True
    ok_bounds = True
    ids = set()
    for r in rows:
        for k in ("episode_id", "mode", "seed", "observable_trace", "reward_by_action"):
            if k not in r:
                ok_schema = False
        rba = r.get("reward_by_action", {})
        for lev in ("H", "W", "H_THEN_W"):
            if lev not in rba:
                ok_schema = False
            elif not (-0.01 <= rba[lev] <= 1.01):   # true_score is capability*honesty in [0,1]
                ok_bounds = False
        ids.add(r.get("episode_id"))
    check("cache/schema_complete", ok_schema, "all rows have required fields + 3 active levers")
    check("cache/reward_in_unit_interval", ok_bounds, "all measured rewards in [0,1]")
    check("cache/unique_episode_ids", len(ids) == len(rows), f"{len(ids)} unique / {len(rows)} rows")

    # meta hash matches the bytes (rebuild the digest the same way build_trace_dataset does)
    meta_path = os.path.join(PROJ, "gpt_oss", "data", "out", "action_outcome_cache.meta.json")
    if os.path.exists(meta_path):
        meta = json.loads(open(meta_path).read())
        blob = "\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()
        digest = hashlib.sha256(blob).hexdigest()[:16]
        check("cache/meta_hash_matches_bytes", digest == meta.get("dataset_sha256_16"),
              f"meta={meta.get('dataset_sha256_16')} recomputed={digest}")


# ---------------------------------------------------------------- 4. train/eval split disjoint
def test_split_disjoint():
    out = os.path.join(PROJ, "gpt_oss", "data", "out")
    for tag, tr, ev in [("SFT", "trace_action_train.jsonl", "trace_action_eval.jsonl"),
                        ("DPO", "dpo_pairs_train.jsonl", "dpo_pairs_eval.jsonl")]:
        ptr, pev = os.path.join(out, tr), os.path.join(out, ev)
        if not (os.path.exists(ptr) and os.path.exists(pev)):
            check(f"split/{tag}_files_exist", False, "rebuild via build_*_dataset.py")
            continue
        tr_seeds = {r["seed"] for r in _load_jsonl(ptr)}
        ev_seeds = {r["seed"] for r in _load_jsonl(pev)}
        check(f"split/{tag}_seeds_disjoint", tr_seeds.isdisjoint(ev_seeds),
              f"train={sorted(tr_seeds)} eval={sorted(ev_seeds)}")

    # the published private eval seeds must be exactly the held-out seeds
    priv = _load_jsonl(os.path.join(PROJ, "sia_task", "data", "private", "measured_outcomes.jsonl"))
    priv_seeds = {int(r["episode_id"].rsplit("_", 1)[1]) for r in priv}
    sft_ev_seeds = {r["seed"] for r in _load_jsonl(os.path.join(out, "trace_action_eval.jsonl"))}
    check("split/private_eval_matches_heldout", priv_seeds == sft_ev_seeds,
          f"private={sorted(priv_seeds)} sft_eval={sorted(sft_ev_seeds)}")


# ---------------------------------------------------------------- 5. gold-definition consistency
def test_gold_definition():
    """The cache stores correct_action = pre-registered ep['correct']; the published task and
    compare_policies use cost_adjusted_best. They must agree on the EVAL seeds (else the headline
    accuracy depends on which gold). Mismatches on TRAIN seeds are surfaced (degenerate episodes)."""
    rows = _load_jsonl(os.path.join(PROJ, "gpt_oss", "data", "out", "action_outcome_cache.jsonl"))
    eval_seeds = set(sorted({r["seed"] for r in rows})[-3:])
    eval_mismatch, train_mismatch = [], []
    for r in rows:
        cab = cost_adjusted_best(r["reward_by_action"])
        cor = r.get("correct_action")
        if cab != cor:
            (eval_mismatch if r["seed"] in eval_seeds else train_mismatch).append(
                (r["episode_id"], cor, cab))
    check("gold/eval_seeds_consistent", not eval_mismatch,
          f"eval mismatches={eval_mismatch}" if eval_mismatch
          else "pre-registered == cost-adjusted-best on all eval episodes")
    if train_mismatch:
        print(f"[NOTE] {len(train_mismatch)} train-seed gold mismatches (degenerate episodes, "
              f"do not affect held-out headline): {train_mismatch}")


def main():
    print("=" * 64)
    print("SIA-Lever robustness / integrity regression")
    print("=" * 64)
    test_evaluator_robustness()
    test_no_private_access()
    test_cache_validation()
    test_split_disjoint()
    test_gold_definition()
    print("=" * 64)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        sys.exit(1)
    print("ALL ROBUSTNESS CHECKS PASS")


if __name__ == "__main__":
    main()
