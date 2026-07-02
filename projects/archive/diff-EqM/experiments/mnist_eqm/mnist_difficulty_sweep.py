"""MNIST inpaint difficulty sweep — can inpainting be made a POSITIVE metacognition result?

Prior finding: MNIST inpaint was null because failures were *confident-wrong* (clean
descent to a plausible-but-wrong digit). The probe keys on descent *instability*, so it
was blind. Hypothesis: as the mask grows the model must hallucinate more structure, and
some failures shift from confident-wrong to *collapse/incoherent* (instability) — which
the probe CAN grip.

Two oracles separate the two failure modes (the crux of the experiment):
  - CLASSIFIER oracle (semantic): completed digit still classifies as the true label.
    Failure = confident-wrong (4 completed as a clean 9). Predicted: probe blind (low AUROC).
  - STRUCTURAL oracle (instability, label-free): the completed digit's ink forms ONE
    connected coherent component. Failure = broken/disconnected/blobby completion.
    Predicted: probe CAN detect (AUROC rises with mask size) -> the positive result.

Action arms per (mask_frac, oracle): vanilla / random-restart (neg) / probe-restart
(treatment) / oracle ceiling (pos), equal NFE (same R draws). Reports AUROC + probe-random
gap vs mask_frac, for both oracles. A positive regime (structural AUROC > chance, gap>0)
confirms the scope-boundary mechanism: instability-failures are detectable even in inpainting.

Run: python mnist_difficulty_sweep.py --run runs/mnist --fracs 0.3 0.45 0.6 0.75 0.9
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from mnist_eqm import MnistEqM, SmallCNN, load_mnist
from mnist_inpaint import make_masks, repaint, consistent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "separability_diagnostic"))
from probe_validate import feature_groups          # noqa: E402
from learned_probe import fit_logreg, auc, within_norm_auc  # noqa: E402


def structural_valid(xt, ink_thr=0.0, min_ink=10):
    """Label-free coherence oracle: completed digit ink is ONE connected component.
    Broken/disconnected/empty completion = structurally invalid (an instability signature)."""
    x = xt.clamp(-1, 1).detach().cpu().numpy()
    out = np.zeros(len(x), bool)
    for i in range(len(x)):
        ink = x[i, 0] > ink_thr
        if ink.sum() < min_ink:
            out[i] = False; continue            # empty/near-empty = incoherent
        _, n = ndimage.label(ink)
        out[i] = (n == 1)                        # single connected stroke = coherent
    return out


def deconf_auc(dn, dd, dv, R):
    """held-out de-confounded AUROC: probe over descent shape predicts INVALID."""
    X = np.concatenate([np.nan_to_num(feature_groups(dn[r], dd[r])["ALL-shape"],
                                      nan=0.0, posinf=0.0, neginf=0.0) for r in range(R)])
    y = np.concatenate([(~dv[r]).astype(float) for r in range(R)])
    ne = np.concatenate([dn[r][:, -1] for r in range(R)])
    if not (0 < y.mean() < 1):
        return float("nan"), float("nan"), X, y, ne
    raws, wns = [], []
    for s in range(5):
        rng = np.random.default_rng(s); idx = rng.permutation(len(y)); nte = int(len(y) * 0.3)
        te, tr = idx[:nte], idx[nte:]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        w, b = fit_logreg((X[tr] - mu) / sd, y[tr], l2=1.0)
        p = 1 / (1 + np.exp(-np.clip((X[te] - mu) / sd @ w + b, -30, 30)))
        raws.append(auc(y[te], p)); wns.append(within_norm_auc(p, y[te], ne[te]))
    return (float(np.mean(raws)) if raws else float("nan"),
            float(np.mean(wns)) if wns else float("nan"), X, y, ne)


def action(dn, dd, dv, R, seed):
    """probe-restart vs random/oracle at equal NFE for a given validity matrix dv (R,M)."""
    X = np.concatenate([np.nan_to_num(feature_groups(dn[r], dd[r])["ALL-shape"],
                                      nan=0.0, posinf=0.0, neginf=0.0) for r in range(R)])
    y = np.concatenate([(~dv[r]).astype(float) for r in range(R)])
    if not (0 < y.mean() < 1):
        v = float(dv[0].mean())
        return {"vanilla": v, "random_restart": v, "probe_restart": v, "oracle_restart": float(dv.any(0).mean())}
    mu = X.mean(0); sd = X.std(0) + 1e-8
    w, b = fit_logreg((X - mu) / sd, y, l2=1.0)
    M = dv.shape[1]; bi = np.arange(M)

    def pinv(r):
        Xr = np.nan_to_num(feature_groups(dn[r], dd[r])["ALL-shape"], nan=0.0, posinf=0.0, neginf=0.0)
        return 1 / (1 + np.exp(-np.clip((Xr - mu) / sd @ w + b, -30, 30)))
    P = np.stack([pinv(r) for r in range(R)])
    rng = np.random.default_rng(seed)
    return {"vanilla": float(dv[0].mean()),
            "random_restart": float(dv[rng.integers(0, R, M), bi].mean()),
            "probe_restart": float(dv[np.argmin(P, 0), bi].mean()),
            "oracle_restart": float(dv.any(0).mean())}


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, Xte, yte = load_mnist(args.root, 1, args.n)
    Xte, yte = Xte.to(dev), yte.to(dev)
    model = MnistEqM(C=args.width).to(dev)
    model.load_state_dict(torch.load(Path(args.run) / "eqm.pt", map_location=dev, weights_only=False)["model"])
    model.eval()
    clf = SmallCNN().to(dev)
    clf.load_state_dict(torch.load(Path(args.run) / "clf.pt", map_location=dev, weights_only=False)["model"])
    clf.eval()

    rows = []
    for frac in args.fracs:
        rng = np.random.default_rng(args.seed)
        mask = make_masks(len(Xte), 28, 28, args.mask, frac, rng).to(dev)
        dn, dd, dv_cls, dv_str = [], [], [], []
        for r in range(args.R):
            torch.manual_seed(args.seed * 100 + r)
            xt, norm, dot = repaint(model, Xte, mask, args.eta, args.steps, log=True, region=args.dyn_region)
            dn.append(norm); dd.append(dot)
            dv_cls.append(consistent(clf, xt, yte))
            dv_str.append(structural_valid(xt))
        dv_cls = np.stack(dv_cls); dv_str = np.stack(dv_str)
        _, wn_cls, *_ = deconf_auc(dn, dd, dv_cls, args.R)
        _, wn_str, *_ = deconf_auc(dn, dd, dv_str, args.R)
        a_cls = action(dn, dd, dv_cls, args.R, args.seed)
        a_str = action(dn, dd, dv_str, args.R, args.seed)
        row = {"mask": args.mask, "frac": frac,
               "classifier_oracle": {"invalid_rate": round(float((~dv_cls).mean()), 3),
                                     "auc_deconf": round(wn_cls, 4),
                                     "probe_minus_random": round(a_cls["probe_restart"] - a_cls["random_restart"], 4),
                                     **{k: round(v, 4) for k, v in a_cls.items()}},
               "structural_oracle": {"invalid_rate": round(float((~dv_str).mean()), 3),
                                     "auc_deconf": round(wn_str, 4),
                                     "probe_minus_random": round(a_str["probe_restart"] - a_str["random_restart"], 4),
                                     **{k: round(v, 4) for k, v in a_str.items()}}}
        rows.append(row)
        print(f"frac={frac}: CLS auc={wn_cls:.3f} gap={row['classifier_oracle']['probe_minus_random']:+.3f} "
              f"inv={row['classifier_oracle']['invalid_rate']:.2f} | "
              f"STR auc={wn_str:.3f} gap={row['structural_oracle']['probe_minus_random']:+.3f} "
              f"inv={row['structural_oracle']['invalid_rate']:.2f}", flush=True)

    # trend: does structural AUROC rise with mask, and is there a positive regime?
    work = [r for r in rows if 0.05 < r["structural_oracle"]["invalid_rate"] < 0.95]
    pos = [r for r in work if r["structural_oracle"]["auc_deconf"] > 0.6
           and r["structural_oracle"]["probe_minus_random"] > 0.01]
    summary = {"run": args.run, "mask": args.mask, "dyn_region": args.dyn_region,
               "steps": args.steps, "eta": args.eta, "R": args.R, "n": int(len(Xte)), "rows": rows,
               "structural_positive_regimes": [r["frac"] for r in pos],
               "verdict": ("POSITIVE inpaint regime found (structural failures probe-detectable)"
                           if pos else "no positive regime — inpaint failures stay probe-invisible")}
    if len(work) >= 2:
        fr = np.array([r["frac"] for r in work]); au = np.array([r["structural_oracle"]["auc_deconf"] for r in work])
        summary["structural_auc_vs_mask_corr"] = round(float(np.corrcoef(fr, au)[0, 1]), 3)
    out = Path(args.out or (Path(args.run) / "difficulty_sweep")); out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/mnist")
    ap.add_argument("--root", default="data")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--mask", default="center", choices=["center", "half", "random"])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.3, 0.45, 0.6, 0.75, 0.9])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dyn-region", default="masked", choices=["full", "masked"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    main(ap.parse_args())
