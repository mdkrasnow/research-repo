"""Aggregate Phase II training logs into Table C (optimizer) and Table D (image).

Reads the `gradient_metrics.jsonl` each run writes (the repo's existing
per-step diagnostic stream, which the BTM branch extends with btm_* fields) and
produces, per arm and per training window:

  Table C  grad norm before/after clip, clip rate, update norm, update/param
           ratio, nonfinite count, gradient noise proxy, throughput, peak memory
  Table D  exact-target MSE / cosine (evaluation-only grad_x phi vs the BTM
           target), energy statistics, FD estimator accuracy in situ

The comparison that matters is G vs D over TRAINING TIME, so every quantity is
reported in early / mid / late windows rather than as a single average -- the
original direct-scalar failure was invisible in the loss and only showed up as a
growing clip rate late in training.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

ARM_OF = {
    "btm_vector": "V  vector",
    "btm_scalar_exact": "G  scalar exact",
    "btm_scalar_action_exact": "A  action exact",
    "btm_scalar_fd_directional": "D  FD directional",
    "btm_scalar_fd_action": "F  FD action",
}


def load_run(path):
    """Return (meta, rows) for one gradient_metrics.jsonl."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    rows = [r for r in rows if "step" in r]
    rows.sort(key=lambda r: r["step"])
    return rows


def windows(rows, n=3):
    """Split by step into n equal windows: early / mid / late."""
    if not rows:
        return []
    lo, hi = rows[0]["step"], rows[-1]["step"]
    if hi == lo:
        return [("all", rows)]
    edges = [lo + (hi - lo) * i / n for i in range(n + 1)]
    names = ["early", "mid", "late"] if n == 3 else [f"w{i}" for i in range(n)]
    out = []
    for i in range(n):
        sel = [r for r in rows
               if edges[i] <= r["step"] <= edges[i + 1]]
        if sel:
            out.append((names[i], sel))
    return out


def _f(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] is not None
            and isinstance(r[key], (int, float)) and not math.isnan(r[key])]
    return vals


def summarize(rows, tag, arm, seed):
    recs = []
    for wname, w in windows(rows):
        gn = _f(w, "grad_norm")
        ugn = _f(w, "unclipped_grad_norm") or gn
        clipped = _f(w, "clipped")
        dtn = _f(w, "delta_theta_norm")
        pn = _f(w, "param_norm")
        rec = {
            "run": tag, "arm": arm, "seed": seed, "window": wname,
            "steps": f"{w[0]['step']}-{w[-1]['step']}", "n": len(w),
            "grad_norm_median": st.median(gn) if gn else float("nan"),
            "grad_norm_p95": (sorted(gn)[int(0.95 * (len(gn) - 1))]
                              if gn else float("nan")),
            "unclipped_median": st.median(ugn) if ugn else float("nan"),
            "unclipped_max": max(ugn) if ugn else float("nan"),
            "clip_rate_pct": (100.0 * sum(bool(c) for c in clipped) / len(clipped)
                              if clipped else float("nan")),
            "delta_theta_median": st.median(dtn) if dtn else float("nan"),
            "param_norm_median": st.median(pn) if pn else float("nan"),
        }
        if dtn and pn:
            rec["update_over_param"] = rec["delta_theta_median"] / (
                rec["param_norm_median"] + 1e-30)
        for k in ("target_cosine", "target_mse_per_dim", "target_norm_ratio",
                  "E_mean", "E_std", "field_norm", "fd_h_mean", "fd_gap_abs",
                  "target_cosine_near_data", "target_cosine_far",
                  "probe_delta_L"):
            v = _f(w, k)
            if v:
                rec[k] = st.median(v)
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+",
                    help="dirs to search for gradient_metrics.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    allrecs = []
    for root in args.roots:
        for p in glob.glob(os.path.join(root, "**", "gradient_metrics.jsonl"),
                           recursive=True):
            tag = os.path.basename(os.path.dirname(os.path.dirname(p)))
            arm, seed = "?", "?"
            for k in ARM_OF:
                if k in p or k in tag:
                    arm = k
            if "_s" in tag:
                seed = tag.rsplit("_s", 1)[-1][:1]
            rows = load_run(p)
            if rows:
                allrecs += summarize(rows, tag, arm, seed)

    if not allrecs:
        print("no gradient_metrics.jsonl found under: " + ", ".join(args.roots))
        return

    print("\n## Table C — optimizer behaviour by training window\n")
    print("| arm | seed | window | steps | grad norm (med) | grad p95 | "
          "unclipped max | clip rate % | Δθ (med) | Δθ/‖θ‖ |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(allrecs, key=lambda r: (r["arm"], r["seed"], r["steps"])):
        print(f"| {ARM_OF.get(r['arm'], r['arm'])} | {r['seed']} | {r['window']} "
              f"| {r['steps']} | {r['grad_norm_median']:.4g} "
              f"| {r['grad_norm_p95']:.4g} | {r['unclipped_max']:.4g} "
              f"| {r['clip_rate_pct']:.2f} | {r.get('delta_theta_median', float('nan')):.3g} "
              f"| {r.get('update_over_param', float('nan')):.3g} |")

    print("\n## Table D — learned field vs the corrected BTM target "
          "(evaluation-only exact grad_x phi)\n")
    print("| arm | seed | window | target cosine | cos near-data | cos far | "
          "target MSE/dim | norm ratio | E mean | E std | FD h | FD gap |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(allrecs, key=lambda r: (r["arm"], r["seed"], r["steps"])):
        if "target_cosine" not in r:
            continue
        g = lambda k: r.get(k, float("nan"))
        print(f"| {ARM_OF.get(r['arm'], r['arm'])} | {r['seed']} | {r['window']} "
              f"| {g('target_cosine'):.4f} | {g('target_cosine_near_data'):.4f} "
              f"| {g('target_cosine_far'):.4f} | {g('target_mse_per_dim'):.4g} "
              f"| {g('target_norm_ratio'):.3f} | {g('E_mean'):.4g} "
              f"| {g('E_std'):.4g} | {g('fd_h_mean'):.3g} | {g('fd_gap_abs'):.3g} |")

    # arm-level aggregation of the two quantities the hypothesis turns on
    print("\n## Key mechanistic comparison — clip rate and target cosine, "
          "late window, mean ± std over seeds\n")
    by = defaultdict(list)
    for r in allrecs:
        if r["window"] == "late":
            by[r["arm"]].append(r)
    print("| arm | seeds | late clip rate % | late target cosine |")
    print("|---|---|---|---|")
    for arm, rs in sorted(by.items()):
        cr = [r["clip_rate_pct"] for r in rs if not math.isnan(r["clip_rate_pct"])]
        tc = [r["target_cosine"] for r in rs if "target_cosine" in r]
        def ms(v):
            if not v:
                return "n/a"
            return (f"{st.mean(v):.4g} ± {st.stdev(v):.3g}" if len(v) > 1
                    else f"{v[0]:.4g}")
        print(f"| {ARM_OF.get(arm, arm)} | {len(rs)} | {ms(cr)} | {ms(tc)} |")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(allrecs, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
