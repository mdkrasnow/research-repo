"""Phase 2 (offline) — truncated-trajectory probe.

The validated probe reads the FULL 250-step descent shape. An ONLINE adaptive
sampler must decide at a partial step k_dec < N, using ONLY steps [0:k_dec] — no
peeking at the final state. Before spending any GPU, answer from the CACHED pool:

  how early is failure detectable? i.e. held-out de-confounded AUROC vs k_dec.

If AUROC stays >~0.75 down to k_dec ~ 0.6N, an online sampler is worth building:
there is enough early signal to reallocate compute causally. If it collapses to
the norm floor as soon as we truncate, online detection is dead -> stick with
post-hoc best-of-R (already validated).

Reuses the cached Stage-1 trajectories + the SAME feature builder + numpy LR.
For each k_dec, also SAVE a deployable partial probe artifact for the online
sampler. CPU, seconds.

Run: python partial_probe.py --folder runs/b2_vanilla
"""
import argparse
import json
from pathlib import Path

import numpy as np

from probe_validate import feature_groups, load
from learned_probe import auc, within_norm_auc, fit_logreg


def heldout(X, y, norm_end, frac=0.30, seeds=5, l2=1.0):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    raws, wns = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(y)); nte = int(len(y) * frac)
        te, tr = idx[:nte], idx[nte:]
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        w, b = fit_logreg((X[tr] - mu) / sd, y[tr], l2=l2)
        pte = 1.0 / (1.0 + np.exp(-np.clip((X[te] - mu) / sd @ w + b, -30, 30)))
        raws.append(auc(y[te], pte))
        wns.append(within_norm_auc(pte, y[te], norm_end[te]))
    return np.mean(raws), np.std(raws), np.mean(wns), np.std(wns)


def main(args):
    folder = Path(args.folder)
    norm, dot, y = load(folder)               # full-length (N, T)
    T = norm.shape[1]
    norm_end_full = norm[:, -1]
    fracs = [float(x) for x in args.kfracs.split(",")]
    out = folder / "results" / "partial_probe"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    lines = ["EqM PARTIAL-trajectory probe — how early is failure detectable?",
             "=" * 60,
             f"good={int((y==0).sum())} garbage={int((y==1).sum())} full_T={T}",
             "k_frac  k_dec  dim   raw_AUROC        within-norm (de-conf)"]
    print("\n".join(lines), flush=True)
    best_for_deploy = None
    for kf in fracs:
        k = max(8, int(round(kf * T)))
        k = min(k, T)
        Xk = feature_groups(norm[:, :k], dot[:, :k])["ALL-shape"]
        # de-confound against the norm AT THE DECISION STEP (causal), not the end
        norm_dec = norm[:, k - 1]
        r, rs, wn, wns = heldout(Xk, y, norm_dec)
        rows.append((kf, k, Xk.shape[1], r, rs, wn, wns))
        line = f"{kf:5.2f}  {k:4d}  {Xk.shape[1]:3d}   {r:.3f}±{rs:.3f}     {wn:.3f}±{wns:.3f}"
        lines.append(line); print(line, flush=True)
        # save deployable artifact for this k (train on all)
        Xall = np.nan_to_num(Xk, nan=0.0, posinf=0.0, neginf=0.0)
        mu = Xall.mean(0); sd = Xall.std(0) + 1e-8
        w, b = fit_logreg((Xall - mu) / sd, y, l2=args.l2)
        np.savez(out / f"partial_probe_k{k}.npz", w=w, b=np.float64(b), mu=mu, sd=sd,
                 k_dec=np.int64(k), full_T=np.int64(T),
                 feature_spec=json.dumps({"groups": ["oscillation", "slopes", "norm_curve", "dot_curve"],
                                          "down_nn": 16, "down_dd": 8, "truncate": int(k)}))

    # full-length reference (k=T) for the ceiling
    full = next((row for row in rows if row[1] == T), None)
    # decision: earliest k whose de-conf within-norm AUROC >= action-ish 0.75
    actionable = [row for row in rows if np.isfinite(row[5]) and row[5] >= args.bar]
    earliest = min(actionable, key=lambda r: r[1]) if actionable else None

    if earliest is not None and earliest[0] <= 0.8:
        verdict = (f"ONLINE-VIABLE: de-confounded AUROC >= {args.bar} as early as "
                   f"k_frac={earliest[0]:.2f} (step {earliest[1]}/{T}) -> enough early "
                   f"signal to build an equal-NFE online adaptive sampler.")
        deploy_k = earliest[1]
    elif full is not None and np.isfinite(full[5]) and full[5] >= args.bar:
        verdict = (f"LATE-ONLY: signal only reaches {args.bar} near the end "
                   f"(k_frac~1.0). Online early-restart marginal; post-hoc best-of-R "
                   f"remains the better lever.")
        deploy_k = full[1]
    else:
        verdict = (f"WEAK: truncated de-confounded AUROC stays < {args.bar} everywhere "
                   f"(best {max((r[5] for r in rows if np.isfinite(r[5])), default=float('nan')):.3f}). "
                   f"Early online detection not supported; keep post-hoc rejection.")
        deploy_k = T

    lines += ["", f"action bar (de-conf within-norm) = {args.bar}",
              f"deploy k_dec = {deploy_k} (artifact partial_probe_k{deploy_k}.npz)",
              "", f"## VERDICT: {verdict}"]
    (out / "PARTIAL_PROBE_SUMMARY.md").write_text("\n".join(lines) + "\n")
    summary = {"full_T": T, "bar": args.bar, "deploy_k": int(deploy_k),
               "rows": [{"k_frac": r[0], "k_dec": r[1], "raw": round(float(r[3]), 4),
                         "raw_sd": round(float(r[4]), 4), "within_norm": round(float(r[5]), 4),
                         "within_norm_sd": round(float(r[6]), 4)} for r in rows],
               "verdict": verdict}
    (out / "partial_probe.json").write_text(json.dumps(summary, indent=2))
    print(f"\n## VERDICT: {verdict}", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--kfracs", default="0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--bar", type=float, default=0.75)
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()
    main(args)
