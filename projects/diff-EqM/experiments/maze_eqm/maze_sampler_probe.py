"""Maze-EqM sampler-budget probe.

Loads a trained maze-EqM and sweeps (steps, eta) reporting vanilla valid-rate on a
dataset. Purpose: find the descent budget where vanilla solves a workable fraction
(~0.2-0.8) so the metacognition best-of-R has an invalid-band to act in. No training.

The GPU scale-up sbatch hardcoded steps=25 eta=0.01 (budget 0.25) which under-descends
c10 OOD -> vanilla ~0 valid -> no signal. This finds the right budget on the saved model.

Run: python maze_sampler_probe.py --ckpt runs/maze_gpu_s1/model.pt \
     --data data/maze_c10_ood.npz --n 400
"""
import argparse
import json

import torch

from eqm_maze import MazeEqM, load, gd_sample
from maze_metacog import valid_mask


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    W = int(ck.get("args", {}).get("width", 64))
    m = MazeEqM(C=W).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    cond, _ = load(args.data); cond = cond[:args.n].to(dev)

    grid = [(s, e) for e in args.etas for s in args.steps_list]
    rows = []
    for steps, eta in grid:
        torch.manual_seed(0)
        xt, _, _ = gd_sample(m, cond, eta, steps, log=False)
        vr = float(valid_mask(xt.cpu().numpy(), cond).mean())
        rows.append({"steps": steps, "eta": eta, "budget": round(steps * eta, 3),
                     "vanilla_valid": round(vr, 4)})
        print(f"  steps={steps:4d} eta={eta:.3f} budget={steps*eta:.2f}  vanilla_valid={vr:.3f}",
              flush=True)
    # pick the point closest to the middle of the workable band
    band = [r for r in rows if 0.2 <= r["vanilla_valid"] <= 0.8]
    pick = min(band, key=lambda r: abs(r["vanilla_valid"] - 0.5)) if band else \
        max(rows, key=lambda r: r["vanilla_valid"])
    out = {"ckpt": args.ckpt, "data": args.data, "width": W, "n": int(cond.shape[0]),
           "grid": rows, "pick": pick,
           "verdict": "workable band found" if band else "NO workable band — model/data too hard"}
    print(json.dumps(out, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/maze_c10_ood.npz")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--steps-list", type=int, nargs="+", dest="steps_list",
                    default=[25, 40, 80, 150, 250])
    ap.add_argument("--etas", type=float, nargs="+", default=[0.02, 0.03])
    ap.add_argument("--out", default="")
    main(ap.parse_args())
