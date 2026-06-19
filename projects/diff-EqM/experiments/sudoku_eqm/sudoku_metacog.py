"""Sudoku-EqM trajectory-metacognition — does the descent-shape probe catch
constraint-reasoning failures, and does probe-restart beat random at equal NFE?

Mirrors maze_metacog.py exactly (same probe, same equal-NFE best-of-R arms, same
de-confounded held-out AUROC) but on the Sudoku-EqM with an EXACT constraint oracle.
This is the third task type (CSP) for the cross-task scope map.

Arms (equal NFE, same R draws): vanilla / random-restart (neg) / probe-restart
(treatment) / oracle (pos). Success = probe-restart > random at equal NFE.

Run: python sudoku_metacog.py --ckpt runs/sudoku/model.pt --data data/sudoku_test.npz \
       --steps 200 --eta 0.02 --R 4 --n 800
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from sudoku_eqm import SudokuEqM, load, gd_sample, valid_mask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "separability_diagnostic"))
from probe_validate import feature_groups          # noqa: E402
from learned_probe import fit_logreg, auc, within_norm_auc  # noqa: E402


def heldout_auc(X, y, ne, frac=0.3, seeds=5, l2=1.0):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    raws, wns = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s); idx = rng.permutation(len(y)); nte = int(len(y) * frac)
        te, tr = idx[:nte], idx[nte:]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        w, b = fit_logreg((X[tr] - mu) / sd, y[tr], l2=l2)
        p = 1 / (1 + np.exp(-np.clip((X[te] - mu) / sd @ w + b, -30, 30)))
        raws.append(auc(y[te], p)); wns.append(within_norm_auc(p, y[te], ne[te]))
    return (float(np.mean(raws)) if raws else float("nan"),
            float(np.mean(wns)) if wns else float("nan"))


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    W = int(ck.get("args", {}).get("width", 96))
    cond, _, sols, masks = load(args.data)
    Nd = sols.shape[-1]                              # grid size N from data
    m = SudokuEqM(cond_ch=cond.shape[1], C=W, D=Nd).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    cond = cond[:args.n].to(dev); sols = sols[:args.n]; masks = masks[:args.n]
    M = cond.shape[0]

    dn, dd, dv = [], [], []
    for r in range(args.R):
        torch.manual_seed(args.seed * 100 + r)
        xt, norm, dot = gd_sample(m, cond, args.eta, args.steps, log=True)
        dn.append(norm); dd.append(dot)
        dv.append(valid_mask(xt.cpu().numpy(), sols, masks))
        print(f"  draw {r}: solve={dv[-1].mean():.3f}", flush=True)
    dv = np.stack(dv)

    Xall = np.concatenate([feature_groups(dn[r], dd[r])["ALL-shape"] for r in range(args.R)])
    yall = np.concatenate([(~dv[r]).astype(float) for r in range(args.R)])
    neall = np.concatenate([dn[r][:, -1] for r in range(args.R)])
    inv_rate = float(yall.mean())
    raw_auc, wn_auc = heldout_auc(Xall, yall, neall)

    Xn = np.nan_to_num(Xall, nan=0.0, posinf=0.0, neginf=0.0)
    mu = Xn.mean(0); sd = Xn.std(0) + 1e-8
    w, b = fit_logreg((Xn - mu) / sd, yall, l2=1.0)

    def p_invalid(norm, dot):
        X = np.nan_to_num(feature_groups(norm, dot)["ALL-shape"], nan=0.0, posinf=0.0, neginf=0.0)
        return 1 / (1 + np.exp(-np.clip((X - mu) / sd @ w + b, -30, 30)))

    pinv = np.stack([p_invalid(dn[r], dd[r]) for r in range(args.R)])
    bi = np.arange(M)
    vanilla = float(dv[0].mean())
    probe = float(dv[np.argmin(pinv, 0), bi].mean())
    rng = np.random.default_rng(args.seed)
    random_ = float(dv[rng.integers(0, args.R, M), bi].mean())
    oracle = float(dv.any(0).mean())
    gap = probe - random_; band = oracle - random_
    rec = gap / band if band > 1e-9 else float("nan")
    verdict = ("PROBE>RANDOM at equal NFE — metacognition works on EqM constraint reasoning"
               if gap > 0.01 and probe <= oracle + 1e-9 else
               "PROBE≈RANDOM — dynamics probe not actionable here" if abs(gap) <= 0.01 else
               "PROBE<RANDOM — anti-correlated")
    res = {"data": Path(args.data).name, "M": M, "R": args.R, "steps": args.steps, "eta": args.eta,
           "invalid_rate": round(inv_rate, 4), "probe_raw_auc": round(raw_auc, 4),
           "probe_within_norm_auc": round(wn_auc, 4),
           "solve_rate": {"vanilla": round(vanilla, 4), "random_restart": round(random_, 4),
                          "probe_restart": round(probe, 4), "oracle_restart": round(oracle, 4)},
           "probe_minus_random": round(gap, 4),
           "fraction_oracle_recovered": None if not np.isfinite(rec) else round(float(rec), 3),
           "nfe_per_output": f"R*steps = {args.R}*{args.steps}", "verdict": verdict}
    out = Path(args.out or "runs/sudoku_metacog"); out.mkdir(parents=True, exist_ok=True)
    (out / f"metacog_s{args.steps}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/sudoku_test.npz")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    main(ap.parse_args())
