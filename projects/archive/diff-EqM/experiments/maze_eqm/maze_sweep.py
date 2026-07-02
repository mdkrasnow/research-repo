"""Maze-EqM — multi-seed × multi-tier metacognition sweep (CPU, overnight).

Runs maze_metacog across seeds × OOD tiers × operating points, aggregates the
probe-restart gain and probe AUROC with mean ± 95% CI. Turns the 2-seed result into
a proper multi-seed CI result. Pure CPU, no cluster.

Run: python maze_sweep.py --seeds 5
"""
import argparse
import json
from pathlib import Path

import numpy as np

import maze_metacog as MM


CONFIGS = [
    # tier file, grid tag, steps, eta (operating points chosen for failure headroom)
    ("data/maze_c7_ood.npz", "c7", 20, 0.01),
    ("data/maze_c10_ood.npz", "c10", 25, 0.01),
    ("data/maze_c10_ood.npz", "c10_mid", 15, 0.02),   # ~23% fail (valid headroom; NOT the degenerate eta=0.01 floor)
]


def main(args):
    import types
    out = Path("runs/maze_sweep"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for (data, tag, steps, eta) in CONFIGS:
        for seed in range(args.seeds):
            a = types.SimpleNamespace(ckpt=args.ckpt, data=data, n=args.n, R=args.R,
                                      steps=steps, eta=eta, thr=0.0, seed=seed,
                                      out=f"runs/maze_sweep/{tag}_s{seed}")
            try:
                r = MM.main(a)
            except Exception as e:
                print(f"  [skip] {tag} seed{seed}: {e}", flush=True)
                continue
            rows.append({"tier": tag, "seed": seed, "invalid": r["invalid_rate"],
                         "auc": r["probe_within_norm_auc"],
                         "vanilla": r["valid_rate"]["vanilla"],
                         "random": r["valid_rate"]["random_restart"],
                         "probe": r["valid_rate"]["probe_restart"],
                         "oracle": r["valid_rate"]["oracle_restart"],
                         "gap": r["probe_minus_random"]})
            print(f"  [{tag} s{seed}] auc={r['probe_within_norm_auc']:.3f} "
                  f"gap={r['probe_minus_random']:+.3f}", flush=True)

    # aggregate per tier
    def ci(x):
        x = np.array(x, float)
        m = x.mean(); s = x.std(ddof=1) if len(x) > 1 else 0.0
        return m, (1.96 * s / np.sqrt(len(x)) if len(x) > 1 else float("nan"))

    agg = {}
    for tag in sorted(set(r["tier"] for r in rows)):
        sub = [r for r in rows if r["tier"] == tag]
        gm, gci = ci([r["gap"] for r in sub])
        am, aci = ci([r["auc"] for r in sub])
        pm, _ = ci([r["probe"] for r in sub]); rm, _ = ci([r["random"] for r in sub])
        agg[tag] = {"n_seeds": len(sub), "auc_mean": round(am, 4), "auc_ci": round(aci, 4),
                    "gap_mean": round(gm, 4), "gap_ci": round(gci, 4),
                    "probe_mean": round(pm, 4), "random_mean": round(rm, 4),
                    "all_positive": bool(all(r["gap"] > 0 for r in sub))}

    md = ["# Maze-EqM metacognition — multi-seed sweep", "",
          f"seeds={args.seeds} R={args.R} n={args.n} per config", "",
          "| tier | seeds | invalid | probe AUROC (mean±CI) | probe−random gap (mean±CI) | all+ |",
          "|---|---|---|---|---|---|"]
    for tag, a in agg.items():
        inv = np.mean([r["invalid"] for r in rows if r["tier"] == tag])
        md.append(f"| {tag} | {a['n_seeds']} | {inv:.2f} | {a['auc_mean']:.3f} ± {a['auc_ci']:.3f} | "
                  f"{a['gap_mean']:+.3f} ± {a['gap_ci']:.3f} | {'yes' if a['all_positive'] else 'NO'} |")
    allgap = [r["gap"] for r in rows]
    gm, gci = ci(allgap)
    npos = sum(1 for g in allgap if g > 0)
    md += ["",
           f"- pooled probe−random gap: {gm:+.3f} ± {gci:.3f} over {len(allgap)} runs; "
           f"positive on {npos}/{len(allgap)}",
           "",
           f"## VERDICT: {'CONSISTENT — probe-restart > random across seeds/tiers (pooled CI excl 0)' if gm-gci>0 and npos==len(allgap) else 'MIXED — see per-tier'}"]
    (out / "MAZE_SWEEP_SUMMARY.md").write_text("\n".join(md) + "\n")
    (out / "maze_sweep.json").write_text(json.dumps({"rows": rows, "agg": agg}, indent=2))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/maze_c5/model.pt")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    main(args)
