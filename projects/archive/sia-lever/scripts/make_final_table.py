#!/usr/bin/env python3
"""Build the final comparison table by invoking compare_policies with the latest rollout files."""

import glob
import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def latest(pat):
    m = sorted(glob.glob(os.path.join(PROJ, pat)))
    return m[-1] if m else None


def main():
    cmd = [sys.executable, os.path.join(PROJ, "gpt_oss", "eval", "compare_policies.py")]
    base = latest("results/gpt_oss/base_rollouts_*.jsonl")
    adapter = (latest("results/gpt_oss/sft_rollouts_*.jsonl")
               or latest("results/gpt_oss/dpo_rollouts_*.jsonl"))
    if base:
        cmd += ["--base-rollouts", "results/gpt_oss/base_rollouts_*.jsonl"]
    if adapter:
        cmd += ["--adapter-rollouts", os.path.relpath(adapter, PROJ)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJ, check=True)


if __name__ == "__main__":
    main()
