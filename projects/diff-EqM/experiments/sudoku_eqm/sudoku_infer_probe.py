"""Inference diagnostic for a trained SudokuEqM: is board-acc=0 an inference issue or a
fundamental EqM-can't-solve-Sudoku issue? Loads a checkpoint, sweeps (steps, eta, clamp),
reports CELL accuracy (fraction of cells matching the solution) AND board accuracy. No retrain.

cell-acc ~0.11 = chance (model lost); ~0.95 = close (needs better sampling / constraint enforce);
1.0 board only if all 81 cells right.
"""
import argparse
import json

import numpy as np
import torch

from sudoku_real import (SudokuEqM, load_satnet, load_rrn, to_chw, gd_sample, board_acc)


def cell_acc(field_chw, labels_oh):
    """fraction of cells whose argmax matches the solution. field:(B,9,9,9), labels_oh:(B,9,9,9)."""
    pred = np.argmax(field_chw, axis=1)              # (B,9,9)
    gt = np.argmax(labels_oh.cpu().numpy(), axis=-1)  # (B,9,9)
    return float((pred == gt).mean())


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    model = SudokuEqM(width=a.get("width", 384), attn=not a.get("no_attn", False)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    if args.data == "satnet":
        f, l = load_satnet(args.data_dir); n = len(f); f, l = f[int(n*0.9):], l[int(n*0.9):]
    else:
        f, l = load_rrn(args.data_dir, "test")
    f, l = f[:args.n], l[:args.n]
    cond = to_chw(f).to(dev); clampf = to_chw(f).to(dev)
    print(f"[infer] ckpt board_acc(train-reported)={ck.get('board_acc')} n={len(f)}", flush=True)

    rows = []
    for clamp in ([True, False] if args.both_clamp else [bool(args.clamp)]):
        for steps in args.steps_list:
            for eta in args.etas:
                fld = gd_sample(model, cond, eta, steps,
                                clamp_feats=clampf if clamp else None).cpu().numpy()
                ca = cell_acc(fld, l); ba = float(board_acc(fld, f).mean())
                rows.append({"clamp": clamp, "steps": steps, "eta": eta,
                             "cell_acc": round(ca, 4), "board_acc": round(ba, 4)})
                print(f"  clamp={clamp} steps={steps:4d} eta={eta:.3f}  cell-acc={ca:.3f}  board-acc={ba:.3f}", flush=True)
    best = max(rows, key=lambda r: (r["board_acc"], r["cell_acc"]))
    print("\nBEST:", json.dumps(best), flush=True)
    print(json.dumps({"ckpt": args.ckpt, "rows": rows, "best": best}, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", choices=["satnet", "rrn"], default="satnet")
    ap.add_argument("--data-dir", default="data_real/sudoku")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps-list", type=int, nargs="+", dest="steps_list", default=[100, 300, 1000])
    ap.add_argument("--etas", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    ap.add_argument("--clamp", type=int, default=1)
    ap.add_argument("--both-clamp", action="store_true")
    main(ap.parse_args())
