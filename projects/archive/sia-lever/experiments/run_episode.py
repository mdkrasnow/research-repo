"""
Four-stage lever episode driver. This is THE measured loop.

  Stage 1  prediction-only train, score with verifier_v0 (weak harness)  -> looks solved
  Stage 2  W-only continue training (more prediction reward)             -> cheat persists
  Stage 3  H update: re-score SAME model with verifier_v1 (structural)   -> shortcut detected
  Stage 4  H->W: retrain with structural objective, score with v1        -> real improvement

Money line, MEASURED: stage-4 neg_control_mse > stage-2 neg_control_mse.
(High neg-control MSE == honest: the broken-symmetry task has no real symmetry to exploit.)

Usage:
    python experiments/run_episode.py            # run all four stages, dump table
    python experiments/run_episode.py --steps 1500
"""

import argparse
import copy
import json
import os
import time

import torch

from train import train
from verifier import verifier_v0, verifier_v1


def _fmt(m):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

    torch.manual_seed(args.seed)
    report = {}

    print("== Stage 1: prediction-only train (weak harness v0) ==")
    model = train(steps=args.steps, objective="prediction_only", seed=args.seed)
    report["stage1_v0"] = verifier_v0(model)
    report["stage1_v1"] = verifier_v1(model)   # measured but hidden from the weak harness
    print("  v0:", _fmt(report["stage1_v0"]))
    print("  v1(hidden):", _fmt(report["stage1_v1"]))

    print("== Stage 2: W-only continue (more prediction reward) ==")
    model = train(model=model, steps=args.steps, objective="prediction_only", seed=args.seed + 1)
    report["stage2_v1"] = verifier_v1(model)
    model_after_w_only = copy.deepcopy(model)
    print("  v1:", _fmt(report["stage2_v1"]))

    print("== Stage 3: H update -> re-score SAME model with structural verifier v1 ==")
    # No training. The harness changed; the model is identical to end of stage 2.
    report["stage3_v1"] = verifier_v1(model_after_w_only)
    print("  v1:", _fmt(report["stage3_v1"]), " <- shortcut now visible")

    print("== Stage 4: H->W retrain with structural objective ==")
    model_hw = train(model=copy.deepcopy(model_after_w_only), steps=args.steps,
                     objective="structural", seed=args.seed + 2)
    report["stage4_v1"] = verifier_v1(model_hw)
    print("  v1:", _fmt(report["stage4_v1"]))

    # ---- gate ----
    neg2 = report["stage2_v1"]["neg_control_mse"]
    neg4 = report["stage4_v1"]["neg_control_mse"]
    gate_pass = neg4 > neg2
    report["gate"] = {
        "stage2_neg_control_mse": round(neg2, 4),
        "stage4_neg_control_mse": round(neg4, 4),
        "pass": bool(gate_pass),
        "claim": "W-only kept cheating; H->W broke it" if gate_pass else "NOT shown — debug toy",
    }

    print("\n== GATE ==")
    print(json.dumps(report["gate"], indent=2))

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(args.out, f"episode_{stamp}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
