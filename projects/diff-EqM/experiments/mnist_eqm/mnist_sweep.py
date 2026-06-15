"""MNIST inpainting metacognition — multi-seed × multi-mask sweep (CPU).

Runs mnist_inpaint across masks × seeds, aggregates probe-restart gain + AUROC with
mean ± 95% CI. Standard inpainting (RePaint clamp) on real MNIST; classifier-consistency
oracle. Operating points chosen for failure headroom (from the mask scan).

Run: python mnist_sweep.py --seeds 3
"""
import argparse
import json
import types
from pathlib import Path

import numpy as np

import mnist_inpaint as MI


CONFIGS = [
    ("center", 0.4),    # ~33% fail
    ("half",   0.55),   # ~27% fail
    ("center", 0.55),   # ~65% fail (harder)
]


def main(args):
    out = Path("runs/mnist_sweep"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for (mask, frac) in CONFIGS:
        tag = f"{mask}{frac}"
        for seed in range(args.seeds):
            a = types.SimpleNamespace(run=args.run, root="data", n=args.n, R=args.R,
                                      mask=mask, mask_frac=frac, steps=args.steps, eta=0.02,
                                      width=64, seed=seed, dyn_region=args.dyn_region,
                                      out=f"runs/mnist_sweep/{tag}_s{seed}")
            try:
                r = MI.main(a)
            except Exception as e:
                print(f"  [skip] {tag} s{seed}: {e}", flush=True); continue
            rows.append({"mask": tag, "seed": seed, "invalid": r["invalid_rate"],
                         "auc": r["probe_within_norm_auc"],
                         "vanilla": r["consistency_rate"]["vanilla"],
                         "random": r["consistency_rate"]["random_restart"],
                         "probe": r["consistency_rate"]["probe_restart"],
                         "oracle": r["consistency_rate"]["oracle_restart"],
                         "gap": r["probe_minus_random"]})
            print(f"  [{tag} s{seed}] auc={r['probe_within_norm_auc']:.3f} gap={r['probe_minus_random']:+.3f}", flush=True)

    def ci(x):
        x = np.array(x, float); m = x.mean()
        s = x.std(ddof=1) if len(x) > 1 else 0.0
        return m, (1.96 * s / np.sqrt(len(x)) if len(x) > 1 else float("nan"))

    md = ["# MNIST inpainting metacognition — multi-seed sweep", "",
          f"seeds={args.seeds} R={args.R} n={args.n} steps={args.steps}. RePaint clamp on real MNIST; "
          "classifier-consistency oracle.", "",
          "| mask | seeds | invalid | probe AUROC (mean±CI) | probe−random gap (mean±CI) | all+ |",
          "|---|---|---|---|---|---|"]
    for (mask, frac) in CONFIGS:
        tag = f"{mask}{frac}"; sub = [r for r in rows if r["mask"] == tag]
        if not sub:
            continue
        am, aci = ci([r["auc"] for r in sub]); gm, gci = ci([r["gap"] for r in sub])
        inv = np.mean([r["invalid"] for r in sub]); allp = all(r["gap"] > 0 for r in sub)
        md.append(f"| {tag} | {len(sub)} | {inv:.2f} | {am:.3f} ± {aci:.3f} | "
                  f"{gm:+.3f} ± {gci:.3f} | {'yes' if allp else 'NO'} |")
    allg = [r["gap"] for r in rows]; gm, gci = ci(allg); npos = sum(1 for g in allg if g > 0)
    consistent = np.isfinite(gci) and gm - gci > 0 and npos == len(allg)
    md += ["", f"- pooled probe−random gap: {gm:+.3f} ± {gci:.3f} over {len(allg)} runs; positive {npos}/{len(allg)}",
           "", f"## VERDICT: {'CONSISTENT — probe-restart > random across masks & seeds (pooled CI excl 0)' if consistent else 'MIXED — see per-mask'}"]
    (out / "MNIST_SWEEP_SUMMARY.md").write_text("\n".join(md) + "\n")
    (out / "mnist_sweep.json").write_text(json.dumps({"rows": rows}, indent=2))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/mnist")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--dyn-region", default="masked", choices=["masked", "full"])
    args = ap.parse_args()
    main(args)
