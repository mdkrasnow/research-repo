"""MNIST inpainting rung — Step B: RePaint inpainting + trajectory-metacognition.

Standard inpainting protocol (RePaint-style clamp) on a real dataset (MNIST). The
EqM fills a masked region; the ORACLE is a held-out classifier — an inpaint is
"valid" iff the completed digit still classifies as the true label (classifier-
consistency, the standard EBM-inpainting sanity metric). Failure = the EqM fills a
plausible-but-wrong digit (a real spurious minimum). Trajectory-metacognition: a
probe over the descent dynamics predicts the wrong inpaints, and probe-guided restart
beats random at equal NFE.

Masks (standard): center box / bottom-half / random block, size set for headroom.

Run: python mnist_inpaint.py --mask center --mask-frac 0.5 --steps 60 --R 4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mnist_eqm import MnistEqM, SmallCNN, load_mnist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "separability_diagnostic"))
from probe_validate import feature_groups                      # noqa: E402
from learned_probe import fit_logreg, auc, within_norm_auc     # noqa: E402


def make_masks(B, H, W, kind, frac, rng):
    """mask 1=hidden/inpaint, 0=observed. (B,1,H,W)."""
    m = np.zeros((B, 1, H, W), np.float32)
    s = max(4, int(round(frac * H)))
    if kind == "center":
        y0 = (H - s) // 2; x0 = (W - s) // 2
        m[:, 0, y0:y0 + s, x0:x0 + s] = 1.0
    elif kind == "half":
        m[:, 0, H // 2:, :] = 1.0
    else:  # random block per image
        for i in range(B):
            y0 = rng.integers(0, H - s + 1); x0 = rng.integers(0, W - s + 1)
            m[i, 0, y0:y0 + s, x0:x0 + s] = 1.0
    return torch.tensor(m)


@torch.no_grad()
def repaint(model, known, mask, eta, steps, log=False, region="full"):
    """region: 'full' = descent dynamics over the whole field (captures global
    consistency); 'masked' = over the inpainted region only (sparse)."""
    obs = (mask < 0.5)
    xt = torch.where(obs, known, torch.randn_like(known))
    t0 = torch.zeros(known.shape[0], device=known.device)
    norms, dots = [], []
    for _ in range(steps):
        f = model(xt, t0)
        if log:
            fm = f if region == "full" else f * mask
            norms.append(fm.flatten(1).norm(dim=1)); dots.append((fm * xt).flatten(1).sum(1))
        xt = xt + f * eta
        xt = torch.where(obs, known, xt)
    if log:
        return xt, torch.stack(norms, 1).cpu().numpy(), torch.stack(dots, 1).cpu().numpy()
    return xt


def consistent(clf, xt, y):
    """oracle: inpainted digit classifies as the true label."""
    with torch.no_grad():
        pred = clf(xt.clamp(-1, 1)).argmax(1)
    return (pred == y).cpu().numpy()


def heldout_auc(X, y, ne, frac=0.3, seeds=5, l2=1.0):
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
        raws.append(auc(y[te], p)); wns.append(within_norm_auc(p, y[te], ne[te]))
    return (float(np.mean(raws)) if raws else float("nan"),
            float(np.mean(wns)) if wns else float("nan"))


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    _, _, Xte, yte = load_mnist(args.root, 1, args.n)
    Xte, yte = Xte.to(dev), yte.to(dev)
    M = len(Xte)
    model = MnistEqM(C=args.width).to(dev)
    model.load_state_dict(torch.load(Path(args.run) / "eqm.pt", map_location=dev, weights_only=False)["model"])
    model.eval()
    clf = SmallCNN().to(dev)
    clf.load_state_dict(torch.load(Path(args.run) / "clf.pt", map_location=dev, weights_only=False)["model"])
    clf.eval()
    rng = np.random.default_rng(args.seed)
    mask = make_masks(M, 28, 28, args.mask, args.mask_frac, rng).to(dev)
    known = Xte

    dn, dd, dv = [], [], []
    for r in range(args.R):
        torch.manual_seed(args.seed * 100 + r)
        xt, norm, dot = repaint(model, known, mask, args.eta, args.steps, log=True, region=args.dyn_region)
        dn.append(norm); dd.append(dot); dv.append(consistent(clf, xt, yte))
        print(f"  draw {r}: consistent={dv[-1].mean():.3f}", flush=True)
    dv = np.stack(dv)

    Xall = np.concatenate([feature_groups(dn[r], dd[r])["ALL-shape"] for r in range(args.R)])
    yall = np.concatenate([(~dv[r]).astype(float) for r in range(args.R)])
    neall = np.concatenate([dn[r][:, -1] for r in range(args.R)])
    raw_auc, wn_auc = heldout_auc(Xall, yall, neall)
    Xn = np.nan_to_num(Xall, nan=0.0, posinf=0.0, neginf=0.0)
    mu = Xn.mean(0); sd = Xn.std(0) + 1e-8
    w, b = fit_logreg((Xn - mu) / sd, yall, l2=1.0)

    def p_inv(norm, dot):
        X = np.nan_to_num(feature_groups(norm, dot)["ALL-shape"], nan=0.0, posinf=0.0, neginf=0.0)
        return 1 / (1 + np.exp(-np.clip((X - mu) / sd @ w + b, -30, 30)))

    pinv = np.stack([p_inv(dn[r], dd[r]) for r in range(args.R)])
    bi = np.arange(M)
    vanilla = dv[0].mean()
    probe = dv[np.argmin(pinv, 0), bi].mean()
    random_ = dv[rng.integers(0, args.R, M), bi].mean()
    oracle = dv.any(0).mean()
    gap = float(probe - random_); band = float(oracle - random_)
    rec = gap / band if band > 1e-9 else float("nan")
    verdict = ("PROBE>RANDOM at equal NFE — metacognition rescues EqM inpainting (MNIST)"
               if gap > 0.01 and probe <= oracle + 1e-9
               else "PROBE≈RANDOM" if abs(gap) <= 0.01 else "PROBE<RANDOM")

    res = {"task": "mnist_inpaint", "mask": args.mask, "mask_frac": args.mask_frac,
           "M": M, "R": args.R, "steps": args.steps, "eta": args.eta,
           "invalid_rate": round(float(yall.mean()), 4),
           "probe_raw_auc": round(raw_auc, 4), "probe_within_norm_auc": round(wn_auc, 4),
           "consistency_rate": {"vanilla": round(float(vanilla), 4), "random_restart": round(float(random_), 4),
                                "probe_restart": round(float(probe), 4), "oracle_restart": round(float(oracle), 4)},
           "probe_minus_random": round(gap, 4),
           "fraction_oracle_recovered": None if not np.isfinite(rec) else round(float(rec), 3),
           "nfe_per_output": f"R*steps = {args.R}*{args.steps}", "verdict": verdict}
    out = Path(args.out) if args.out else Path(__file__).parent / "runs" / "mnist_inpaint"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"inpaint_{args.mask}_f{args.mask_frac}_s{args.steps}_seed{args.seed}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/mnist")
    ap.add_argument("--root", default="data")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--mask", default="center", choices=["center", "half", "random"])
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dyn-region", default="full", choices=["full", "masked"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    main(args)
