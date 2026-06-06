#!/usr/bin/env python3
"""Render every gpt-oss demo figure on CPU using SYNTHETIC base+LoRA rollouts.

Purpose: so the demo deck has illustrative figures BEFORE GPU access, and so the whole figure +
diagnostics pipeline is exercised/tested. Outputs go to results/gpt_oss/preview/ and titles/files
are clearly marked PREVIEW. The real GPU run (run_gpu_comparison.sh) writes the unprefixed,
authoritative versions under results/gpt_oss/ and results/.

Synthetic policy:
  base  = oracle-sandwich rule but with realistic mistakes (mis-attributes some bad_verifier as W,
          and emits one invalid-JSON response) — i.e., a plausible un-tuned model.
  lora  = oracle-sandwich rule, near-perfect (one residual error) — a plausible tuned model.
NOTHING here is a real gpt-oss measurement; it only demonstrates the figures.
"""

import json
import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "gpt_oss"))
sys.path.insert(0, os.path.join(PROJ, "methods"))
from lever_io import load_cache, cost_adjusted_best, action_json  # noqa: E402
import oracle_sandwich_selector as rule  # noqa: E402

PREVIEW = os.path.join(PROJ, "results", "gpt_oss", "preview")


def synth_rollouts():
    cache = load_cache()
    ev = [r for r in cache if r["seed"] in (7, 8, 9)]
    base, lora = [], []
    for i, ep in enumerate(ev):
        gold = cost_adjusted_best(ep["reward_by_action"])
        rule_act = rule.select(ep["observable_trace"])
        # base: degrade — flip one bad_verifier to W, make one response invalid JSON
        b_act, b_valid, b_raw = rule_act, True, action_json(rule_act, "rule")
        if ep["mode"] == "bad_verifier" and i % 2 == 0:
            b_act, b_raw = "W", action_json("W", "train it harder")     # wrong: trains on bad harness
        if i == 1:
            b_act, b_valid, b_raw = None, False, "I think you should fix the harness maybe"
        base.append({"episode_id": ep["episode_id"], "mode": ep["mode"], "seed": ep["seed"],
                     "action": b_act, "reason": "synthetic-base", "valid_json": b_valid,
                     "raw_response": b_raw})
        # lora: near-perfect (rule), one residual miss
        l_act = rule_act if i != 4 else ("W" if rule_act != "W" else "H")
        lora.append({"episode_id": ep["episode_id"], "mode": ep["mode"], "seed": ep["seed"],
                     "action": l_act, "reason": "synthetic-lora", "valid_json": True,
                     "raw_response": action_json(l_act, "lora")})
    os.makedirs(PREVIEW, exist_ok=True)
    bp = os.path.join(PREVIEW, "base_rollouts_PREVIEW.jsonl")
    lp = os.path.join(PREVIEW, "sft_rollouts_PREVIEW.jsonl")
    for path, rows in [(bp, base), (lp, lora)]:
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return bp, lp


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJ, check=True)


def main():
    bp, lp = synth_rollouts()
    py = sys.executable
    run([py, "gpt_oss/eval/eval_selector.py", "--rollouts", bp, "--tag", "PREVIEW_base", "--out", PREVIEW])
    run([py, "gpt_oss/eval/eval_selector.py", "--rollouts", lp, "--tag", "PREVIEW_lora", "--out", PREVIEW])
    run([py, "gpt_oss/eval/eval_adapter.py", "--adapter-rollouts", lp, "--base-rollouts", bp,
         "--tag", "PREVIEW", "--out", PREVIEW])
    run([py, "gpt_oss/eval/compare_policies.py", "--base-rollouts", bp, "--adapter-rollouts", lp,
         "--out-dir", PREVIEW, "--title-tag", "[PREVIEW/synthetic]"])
    print(f"\nPREVIEW figures (synthetic, illustrative) -> {PREVIEW}")
    print("Replace with the real GPU run: bash scripts/run_gpu_comparison.sh")


if __name__ == "__main__":
    main()
