"""v11 Metacognitive Curriculum — probe-guided hard-example mining for EqM training.

Per metacog_curriculum_proposal.md. Continues from a base SudokuEqM checkpoint; every K
epochs runs a MINING PASS (the model's own GD sampling on a train pool), scores each example
by a selector, and reweights the base EqM loss toward the hard ones. Tracks CELL-ACCURACY on
held-out test (the real signal — loss decouples from sampling quality, see SATNet v3).

Four arms, SAME base + SAME compute (controls per CLAUDE.md):
  uniform  — no mining (floor; this is the over-training-degrades baseline)
  random   — reweight a random equal-size subset (floor; isolates "is it the FAILURE selection")
  oracle   — reweight true-hard examples (high cell-error) (ceiling)
  probe    — reweight probe-flagged examples (descent-shape probe predicts hard) (TREATMENT)

Success = probe holds/improves cell-acc vs random, and ≈ oracle. Then the probe enables
hard-mining without labels (the no-label payoff).

Run: python sudoku_curriculum.py --base runs/sudoku_real/satnet_w384_v2.pt \
       --data-dir data_real/sudoku --arms uniform random oracle probe --rounds 8 --k 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sudoku_real import (SudokuEqM, load_satnet, to_chw, eqm_target, gd_sample, board_acc)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "separability_diagnostic"))
from probe_validate import feature_groups          # noqa: E402
from learned_probe import fit_logreg               # noqa: E402


def cell_err_per_ex(fld, labels_oh):
    """per-example fraction of cells wrong. fld:(B,9,9,9), labels_oh:(B,9,9,9)."""
    pred = np.argmax(fld, 1); gt = np.argmax(labels_oh.cpu().numpy(), -1)
    return 1.0 - (pred == gt).reshape(len(pred), -1).mean(1)


def run_arm(arm, base_sd, x1_tr, cond_tr, x1_pool, cond_pool, lab_pool, clamp_pool,
            cond_te, fte, lte, args, dev):
    torch.manual_seed(args.seed)
    model = SudokuEqM(width=args.width, attn=not args.no_attn).to(dev)
    model.load_state_dict(base_sd);
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    P = cond_pool.shape[0]; bs = args.batch
    traj = []
    rng = np.random.default_rng(args.seed)
    for rnd in range(args.rounds):
        # ---- mining pass: the model's own GD sampling on the pool ----
        model.eval()
        fld, norm, dot = gd_sample(model, cond_pool, args.eta, args.steps,
                                   clamp_feats=clamp_pool, log=True)
        fld = fld.cpu().numpy()
        cerr = cell_err_per_ex(fld, lab_pool)                       # (P,)
        med = np.median(cerr)
        # ---- selector -> per-example weight w in [1, 1+lam] ----
        if arm == "uniform":
            w = np.ones(P, np.float32)
        elif arm == "random":
            w = 1.0 + args.lam * (rng.random(P) < 0.5).astype(np.float32)
        elif arm == "oracle":
            w = 1.0 + args.lam * (cerr > med).astype(np.float32)
        elif arm == "probe":
            X = np.nan_to_num(feature_groups(norm, dot)["ALL-shape"], nan=0.0, posinf=0.0, neginf=0.0)
            y = (cerr > med).astype(np.float32)
            if 0 < y.mean() < 1:
                mu = X.mean(0); sd = X.std(0) + 1e-8
                pw, pb = fit_logreg((X - mu) / sd, y, l2=1.0)
                s = 1 / (1 + np.exp(-np.clip((X - mu) / sd @ pw + pb, -30, 30)))
                w = 1.0 + args.lam * s.astype(np.float32)
            else:
                w = np.ones(P, np.float32)
        wt = torch.tensor(w, device=dev)
        # ---- K weighted-EqM-loss epochs on the pool ----
        model.train()
        for _ in range(args.k):
            perm = torch.randperm(P, device=dev)
            for i in range(0, P, bs):
                idx = perm[i:i + bs]
                t = torch.rand(len(idx), device=dev)
                xt, target = eqm_target(x1_pool[idx], t)
                pred = model(xt, t, cond_pool[idx])
                pe = ((pred - target) ** 2).mean(dim=[1, 2, 3])     # per-example MSE
                loss = (wt[idx] * pe).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
        # ---- eval cell-acc + board-acc on held-out test ----
        model.eval()
        tf = gd_sample(model, cond_te, args.eta, args.steps, clamp_feats=clamp_te(args, fte, dev)).cpu().numpy()
        cacc = float((np.argmax(tf, 1) == np.argmax(lte.numpy(), -1)).mean())
        bacc = float(board_acc(tf, fte).mean())
        # hardness of the mined set this round (should drop if learning fixes failures)
        traj.append({"round": rnd, "mined_frac_hard": float((cerr > med).mean()),
                     "pool_cell_err": round(float(cerr.mean()), 4),
                     "test_cell_acc": round(cacc, 4), "test_board_acc": round(bacc, 4)})
        print(f"  [{arm}] round {rnd}: pool_err={cerr.mean():.3f} test_cell_acc={cacc:.3f} board={bacc:.3f}", flush=True)
    return traj


def clamp_te(args, fte, dev):
    return to_chw(fte).to(dev) if args.clamp else None


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    f, l = load_satnet(args.data_dir); n = len(f); ntr = int(n * 0.9)
    ftr, ltr, fte, lte = f[:ntr], l[:ntr], f[ntr:ntr + args.eval_n], l[ntr:ntr + args.eval_n]
    base = torch.load(args.base, map_location=dev, weights_only=False)
    print(f"[v11 curriculum] base board_acc={base.get('board_acc')} train={len(ftr)} pool={args.pool} "
          f"arms={args.arms} lam={args.lam} K={args.k} rounds={args.rounds}", flush=True)

    cond_tr = to_chw(ftr).to(dev); x1_tr = to_chw(ltr).to(dev)
    # mining pool = first P train examples
    fp, lp = ftr[:args.pool], ltr[:args.pool]
    cond_pool = to_chw(fp).to(dev); x1_pool = to_chw(lp).to(dev)
    clamp_pool = to_chw(fp).to(dev) if args.clamp else None
    cond_te = to_chw(fte).to(dev)

    results = {}
    t0 = time.time()
    for arm in args.arms:
        print(f"=== ARM {arm} ({time.time()-t0:.0f}s) ===", flush=True)
        results[arm] = run_arm(arm, base["model"], x1_tr, cond_tr, x1_pool, cond_pool, lp,
                               clamp_pool, cond_te, fte, lte, args, dev)
    # summary: final test_cell_acc per arm + deltas
    fin = {a: results[a][-1]["test_cell_acc"] for a in args.arms}
    summary = {"base": args.base, "base_board_acc": base.get("board_acc"),
               "final_test_cell_acc": fin, "trajectories": results,
               "probe_minus_random": round(fin.get("probe", 0) - fin.get("random", 0), 4) if "probe" in fin and "random" in fin else None,
               "probe_minus_uniform": round(fin.get("probe", 0) - fin.get("uniform", 0), 4) if "probe" in fin and "uniform" in fin else None,
               "oracle_minus_random": round(fin.get("oracle", 0) - fin.get("random", 0), 4) if "oracle" in fin and "random" in fin else None}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "curriculum.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data-dir", default="data_real/sudoku")
    ap.add_argument("--arms", nargs="+", default=["uniform", "random", "oracle", "probe"])
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--k", type=int, default=5, help="weighted epochs per round")
    ap.add_argument("--lam", type=float, default=1.0, help="max mining upweight")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--clamp", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/sudoku_curriculum")
    main(ap.parse_args())
