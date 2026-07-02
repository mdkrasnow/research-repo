"""Inpainting / repair rung — RePaint-style inpainting on the trained maze-EqM (CPU).

Capability test: given a maze solution with a masked window, can the EqM inpaint the
missing path segment consistent with the observed path + walls — and does trajectory-
metacognition rescue failed inpaints at equal NFE? Exact BFS labels, real trained EqM,
NO retraining (RePaint-style observed-clamp uses the Step-2 path model directly).

Inference (RePaint): GD-sample the path field; after each step overwrite the OBSERVED
cells with the known path (hard constraint), let the model fill the masked window.
Failure = the inpainted segment breaks connectivity (spurious minimum) -> BFS-invalid.

Arms (best-of-R, equal NFE, same R draws): vanilla(draw0)/random-restart/probe-restart/
oracle-restart. Metric = inpaint-valid-rate (completed path passes BFS).

Run: python maze_inpaint.py --ckpt runs/maze_c5/model.pt --data data/maze_c5_test.npz \
     --mask-frac 0.5 --steps 60 --eta 0.02 --R 4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eqm_maze import MazeEqM, load
from gen_maze_data import path_valid, decode_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "separability_diagnostic"))
from probe_validate import feature_groups                      # noqa: E402
from learned_probe import fit_logreg, auc, within_norm_auc     # noqa: E402


def make_masks(cond, target, rng, frac, kind="band"):
    """mask (1=hidden/inpaint, 0=observed). kinds:
       patch = square patch on a random path cell (easy: connects 2 known endpoints).
       band  = full horizontal strip of height frac*H centred on a random path cell
               (hard: path must be reconstructed across decoy wall-gaps).
       multi = two disjoint patches (medium)."""
    M, _, H, W = cond.shape
    masks = np.zeros((M, 1, H, W), np.float32)
    side = max(2, int(round(frac * H)))
    for i in range(M):
        pc = np.argwhere(target[i, 0].cpu().numpy() > 0)
        cy, cx = pc[rng.integers(len(pc))]
        if kind == "band":
            y0 = int(np.clip(cy - side // 2, 0, H - side))
            masks[i, 0, y0:y0 + side, :] = 1.0
        elif kind == "multi":
            for (py, px) in pc[rng.choice(len(pc), size=2, replace=False)]:
                s2 = max(2, side // 2)
                y0 = int(np.clip(py - s2 // 2, 0, H - s2)); x0 = int(np.clip(px - s2 // 2, 0, W - s2))
                masks[i, 0, y0:y0 + s2, x0:x0 + s2] = 1.0
        else:  # patch
            y0 = int(np.clip(cy - side // 2, 0, H - side)); x0 = int(np.clip(cx - side // 2, 0, W - side))
            masks[i, 0, y0:y0 + side, x0:x0 + side] = 1.0
    return torch.tensor(masks)


@torch.no_grad()
def repaint_sample(model, cond, known, mask, eta, steps, log=False):
    """RePaint GD: clamp observed cells to known path each step, fill masked window.
    known:(M,1,H,W) in {-1,+1}; mask:(M,1,H,W) 1=inpaint. Returns final (+dynamics)."""
    dev = cond.device
    obs = (mask < 0.5)
    xt = torch.randn_like(known)
    xt = torch.where(obs, known, xt)
    t0 = torch.zeros(cond.shape[0], device=dev)
    norms, dots = [], []
    for _ in range(steps):
        f = model(xt, t0, cond)
        if log:
            # dynamics over the MASKED (inpainted) region only — that's what varies
            fm = f * mask
            norms.append(fm.flatten(1).norm(dim=1))
            dots.append((fm * xt).flatten(1).sum(1))
        xt = xt + f * eta
        xt = torch.where(obs, known, xt)                        # re-clamp observed
    if log:
        return xt, torch.stack(norms, 1).cpu().numpy(), torch.stack(dots, 1).cpu().numpy()
    return xt


def inpaint_valid(xt, cond, known, mask, thr=0.0):
    """Completed path = observed(known) + generated(masked); BFS validity per maze."""
    M, _, H, W = cond.shape
    obs = (mask < 0.5).cpu().numpy()
    gen = decode_path(xt.cpu().numpy(), thr)                     # (M,1,H,W) {0,1}
    knownb = (known.cpu().numpy() > 0).astype(np.int8)
    out = np.zeros(M, bool)
    for i in range(M):
        full = np.where(obs[i, 0], knownb[i, 0], gen[i, 0]).astype(np.int8)
        wall = cond[i, 0].cpu().numpy().astype(np.int8)
        s = tuple(np.argwhere(cond[i, 1].cpu().numpy() > 0)[0])
        g = tuple(np.argwhere(cond[i, 2].cpu().numpy() > 0)[0])
        out[i] = path_valid(full, wall, s, g)
    return out


def heldout_auc(X, y, norm_end, frac=0.3, seeds=5, l2=1.0):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    raws, wns = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(y)); nte = int(len(y) * frac)
        te, tr = idx[:nte], idx[nte:]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        w, b = fit_logreg((X[tr] - mu) / sd, y[tr], l2=l2)
        p = 1 / (1 + np.exp(-np.clip((X[te] - mu) / sd @ w + b, -30, 30)))
        raws.append(auc(y[te], p)); wns.append(within_norm_auc(p, y[te], norm_end[te]))
    return (float(np.mean(raws)) if raws else float("nan"),
            float(np.mean(wns)) if wns else float("nan"))


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    m = MazeEqM().to(dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    m.load_state_dict(ck["model"]); m.eval()
    cond, tgt = load(args.data)
    cond, tgt = cond[:args.n].to(dev), tgt[:args.n].to(dev)
    M = cond.shape[0]
    rng = np.random.default_rng(args.seed)
    mask = make_masks(cond, tgt, rng, args.mask_frac, args.mask_kind).to(dev)
    known = tgt                                                  # {-1,+1}

    draw_norm, draw_dot, draw_valid = [], [], []
    for r in range(args.R):
        torch.manual_seed(args.seed * 100 + r)
        xt, norm, dot = repaint_sample(m, cond, known, mask, args.eta, args.steps, log=True)
        draw_norm.append(norm); draw_dot.append(dot)
        draw_valid.append(inpaint_valid(xt, cond, known, mask, args.thr))
        print(f"  draw {r}: inpaint-valid={draw_valid[-1].mean():.3f}", flush=True)
    draw_valid = np.stack(draw_valid)

    # detection
    Xall, yall, neall = [], [], []
    for r in range(args.R):
        Xall.append(feature_groups(draw_norm[r], draw_dot[r])["ALL-shape"])
        yall.append((~draw_valid[r]).astype(float)); neall.append(draw_norm[r][:, -1])
    Xall = np.concatenate(Xall); yall = np.concatenate(yall); neall = np.concatenate(neall)
    raw_auc, wn_auc = heldout_auc(Xall, yall, neall)
    Xn = np.nan_to_num(Xall, nan=0.0, posinf=0.0, neginf=0.0)
    mu = Xn.mean(0); sd = Xn.std(0) + 1e-8
    w, b = fit_logreg((Xn - mu) / sd, yall, l2=1.0)

    def p_invalid(norm, dot):
        X = np.nan_to_num(feature_groups(norm, dot)["ALL-shape"], nan=0.0, posinf=0.0, neginf=0.0)
        return 1 / (1 + np.exp(-np.clip((X - mu) / sd @ w + b, -30, 30)))

    pinv = np.stack([p_invalid(draw_norm[r], draw_dot[r]) for r in range(args.R)])
    bi = np.arange(M)
    vanilla = draw_valid[0].mean()
    probe = draw_valid[np.argmin(pinv, 0), bi].mean()
    random_ = draw_valid[rng.integers(0, args.R, M), bi].mean()
    oracle = draw_valid.any(0).mean()
    gap = float(probe - random_); band = float(oracle - random_)
    rec = gap / band if band > 1e-9 else float("nan")
    verdict = ("PROBE>RANDOM at equal NFE — metacognition rescues EqM inpainting"
               if gap > 0.01 and probe <= oracle + 1e-9
               else "PROBE≈RANDOM" if abs(gap) <= 0.01 else "PROBE<RANDOM")

    res = {"task": "maze_inpaint", "data": Path(args.data).name, "grid": int(cond.shape[-1]),
           "M": M, "R": args.R, "mask_frac": args.mask_frac, "steps": args.steps, "eta": args.eta,
           "invalid_rate": round(float(yall.mean()), 4),
           "probe_raw_auc": round(raw_auc, 4), "probe_within_norm_auc": round(wn_auc, 4),
           "valid_rate": {"vanilla": round(float(vanilla), 4), "random_restart": round(float(random_), 4),
                          "probe_restart": round(float(probe), 4), "oracle_restart": round(float(oracle), 4)},
           "probe_minus_random": round(gap, 4),
           "fraction_oracle_recovered": None if not np.isfinite(rec) else round(float(rec), 3),
           "nfe_per_output": f"R*steps = {args.R}*{args.steps}", "verdict": verdict}
    out = Path(args.out) if args.out else Path(__file__).parent / "runs" / "maze_inpaint"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"inpaint_{Path(args.data).stem}_m{args.mask_frac}_s{args.steps}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/maze_c5/model.pt")
    ap.add_argument("--data", default="data/maze_c5_test.npz")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--mask-kind", default="band", choices=["band", "patch", "multi"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    main(args)
