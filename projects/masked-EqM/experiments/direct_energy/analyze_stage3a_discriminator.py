"""
Stage 3A analyzer: turn wfb_stage3a_mechanism_discriminator.py's rows into the
pre-registered classification table, the six diagnostic scatter plots, and an
explicit H1 / H2 / H3 verdict.

The verdict rule is pre-registered in the discriminator's module docstring and
is applied here MECHANICALLY -- it reads the numbers, it does not decide them:

  H1 (Gauss-Newton local model is wrong; repair = LM damping)
      R_B far below 1 AND D_B large at the taken step,
      BUT the infinitesimal transfer d_V < 0 (the direction itself is fine,
      so shortening / damping the step can recover it).

  H2 (local model is right for the wrong minibatch; repair = larger model
      batch or independent acceptance -- damping CANNOT help)
      R_B ~ 1 and D_B << 1 (the model is accurate)
      BUT d_V >= 0 frequently, or trust loss rises even at the SMALLEST eta.
      d_V >= 0 is decisive on its own: by first-order Taylor, if the
      directional derivative of the trust objective along p is non-negative,
      then no step size and no damping of THAT direction can reduce it.

  H3 both.

Warning thresholds (reported as distributions, never as constants to round to):
  D_B > 0.25  or  R_B < 0.5  on a substantial fraction of batches.

Plots (matplotlib, written as PNG; skipped with a notice if unavailable):
  1. D_B vs trust delta L
  2. R_B vs trust delta L
  3. lambda_max vs D_B (fixed batches -- the ONLY valid way to ask whether
     curvature growth tracks declining model fidelity)
  4. source predicted vs actual reduction
  5. source improvement vs independent trust improvement
  6. cross-batch directional alignment C_V, direct vs FBGN
"""
import argparse
import json
import math
import os
from collections import defaultdict


def median(xs):
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def frac(xs, pred):
    xs = [x for x in xs if x is not None]
    return (sum(1 for x in xs if pred(x)) / len(xs)) if xs else None


def bootstrap_ci(xs, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the mean of a paired-difference sample.
    Deterministic (fixed seed); pure-python LCG so numpy is optional."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return (None, None)
    state = seed * 2 + 1
    n = len(xs)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            state = (1103515245 * state + 12345) % (1 << 31)
            s += xs[state % n]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and (abs(x) >= 1e4 or (x != 0 and abs(x) < 1e-3)):
        return f"{x:.3g}"
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="stage3a_rows.jsonl")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.rows))
    os.makedirs(out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.rows)]
    print(f"loaded {len(rows)} rows\n")

    # ---------------- main table: one line per (ckpt, method, lam_mult, eta) --
    key = lambda r: (r["ckpt"], r["method"], r.get("lam_mult"), r["eta_frac"])
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)

    hdr = (f"| {'checkpoint':<10} | {'method':<19} | {'lam_x':>5} | {'eta_frac':>8} | "
           f"{'R_B':>8} | {'D_B':>8} | {'trust desc %':>12} | {'C_V':>9} | "
           f"{'dL_trust':>10} | {'dL_probe':>10} |")
    sep = "|" + "|".join("-" * (len(c) + 2) for c in hdr.split("|")[1:-1]) + "|"
    print("## Stage 3A classification table\n")
    print(hdr)
    print(sep)
    table_lines = [hdr, sep]
    for k in sorted(groups, key=lambda k: (k[0], k[1], k[2] or 0, -k[3])):
        ck, meth, lam_mult, ef = k
        g = groups[k]
        line = (f"| {ck:<10} | {meth:<19} | {str(lam_mult or ''):>5} | {ef:>8.3f} | "
                f"{fmt(median([r['R_B'] for r in g]), 3):>8} | "
                f"{fmt(median([r['D_B'] for r in g]), 4):>8} | "
                f"{fmt(100 * (frac([r['d_V'] for r in g], lambda x: x < 0) or 0), 1):>12} | "
                f"{fmt(median([r['C_V'] for r in g]), 4):>9} | "
                f"{fmt(median([r['delta_L_trust'] for r in g]), 4):>10} | "
                f"{fmt(median([r['delta_L_probe'] for r in g]), 4):>10} |")
        print(line)
        table_lines.append(line)

    # ---------------- the decisive H2 statistic ------------------------------
    print("\n## H2 discriminator: infinitesimal independent-batch transfer d_V = grad L_V . p")
    print("   (eta-INDEPENDENT. d_V >= 0 => NO step size and NO damping can help this direction.)\n")
    verdict_data = {}
    for ck in sorted({r["ckpt"] for r in rows}):
        for meth in sorted({r["method"] for r in rows}):
            sub = [r for r in rows if r["ckpt"] == ck and r["method"] == meth and r["eta_frac"] == 1.0]
            if not sub:
                continue
            dv = [r["d_V"] for r in sub]
            cv = [r["C_V"] for r in sub]
            lo, hi = bootstrap_ci(dv)
            n_desc = sum(1 for x in dv if x < 0)
            print(f"  {ck:<10} {meth:<19} n={len(dv):>2}  descent {n_desc}/{len(dv)}  "
                  f"median d_V={fmt(median(dv))}  mean 95% CI=[{fmt(lo)}, {fmt(hi)}]  "
                  f"median C_V={fmt(median(cv))}")
            verdict_data[(ck, meth)] = {"n": len(dv), "n_descent": n_desc,
                                        "median_d_V": median(dv), "median_C_V": median(cv),
                                        "ci": [lo, hi]}

    # ---------------- eta-floor test: does the SMALLEST step still hurt? -----
    print("\n## Does the trust objective worsen even at the SMALLEST eta tested?")
    print("   (H2's second signature: harm that survives eta -> 0 is a direction defect, not a step-length one.)\n")
    eta_min = min(r["eta_frac"] for r in rows if r["method"] == "fbgn")
    for ck in sorted({r["ckpt"] for r in rows}):
        for meth in sorted({r["method"] for r in rows}):
            sub = [r for r in rows if r["ckpt"] == ck and r["method"] == meth
                   and abs(r["eta_frac"] - eta_min) < 1e-12]
            if not sub:
                continue
            d = [r["delta_L_trust"] for r in sub]
            paired = [r["trust_paired_mean"] for r in sub]
            lo, hi = bootstrap_ci(paired)
            print(f"  {ck:<10} {meth:<19} eta_frac={eta_min:<6.3f} "
                  f"worsened {sum(1 for x in d if x > 0)}/{len(d)}  "
                  f"median dL_V={fmt(median(d))}  paired mean 95% CI=[{fmt(lo)}, {fmt(hi)}]")

    # ---------------- verdict ------------------------------------------------
    print("\n## VERDICT\n")
    for ck in sorted({r["ckpt"] for r in rows}):
        fb = [r for r in rows if r["ckpt"] == ck and r["method"] == "fbgn"]
        if not fb:
            continue
        at1 = [r for r in fb if r["eta_frac"] == 1.0]
        med_R = median([r["R_B"] for r in at1])
        med_D = median([r["D_B"] for r in at1])
        dv = [r["d_V"] for r in at1]
        frac_ascent = frac(dv, lambda x: x >= 0)
        small = [r for r in fb if abs(r["eta_frac"] - eta_min) < 1e-12]
        frac_small_worse = frac([r["delta_L_trust"] for r in small], lambda x: x > 0)

        model_bad = (med_R is not None and med_R < 0.5) or (med_D is not None and med_D > 0.25)
        transfer_bad = (frac_ascent or 0) > 0.25 or (frac_small_worse or 0) > 0.5
        verdict = ("H3 (both local-model failure AND stochastic over-solving)" if model_bad and transfer_bad
                   else "H1 (Gauss-Newton local-model failure) -- repair = LM damping / shorter steps"
                   if model_bad else
                   "H2 (stochastic minibatch over-solving) -- repair = larger model batch / "
                   "independent acceptance; damping CANNOT help"
                   if transfer_bad else
                   "NEITHER fired: FBGN looks locally sound AND transfers -- re-examine the premise")
        print(f"  {ck}: median R_B@eta*={fmt(med_R,3)}  median D_B@eta*={fmt(med_D,4)}  "
              f"frac(d_V>=0)={fmt(frac_ascent,3)}  frac(dL_V>0 at smallest eta)={fmt(frac_small_worse,3)}")
        print(f"    -> {verdict}\n")

    # ---------------- plots --------------------------------------------------
    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[plots] matplotlib unavailable ({e}) -- skipping plots, tables above stand.")
            return 0

        methods = sorted({r["method"] for r in rows})
        colors = {m: c for m, c in zip(methods, ["C0", "C1", "C2", "C3", "C4"])}
        specs = [
            ("D_B", "delta_L_trust", "1_defect_vs_trust", "linearization defect D_B", "trust delta L"),
            ("R_B", "delta_L_trust", "2_R_vs_trust", "reduction ratio R_B", "trust delta L"),
            ("lambda_max", "D_B", "3_lambdamax_vs_defect", "lambda_max (fixed batch)", "linearization defect D_B"),
            ("pred_delta_source", "actual_delta_source", "4_pred_vs_actual", "predicted source reduction", "actual source reduction"),
            ("actual_delta_source", "delta_L_trust", "5_source_vs_trust", "source delta L", "trust delta L"),
        ]
        for xk, yk, name, xl, yl in specs:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for m in methods:
                pts = [(r.get(xk), r.get(yk)) for r in rows if r["method"] == m]
                pts = [(x, y) for x, y in pts if x is not None and y is not None]
                if pts:
                    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=18, alpha=0.7,
                               label=m, color=colors[m])
            if xk == "lambda_max":
                ax.set_xscale("log")
            ax.axhline(0, lw=0.8, color="k", alpha=0.5)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.legend(fontsize=7)
            ax.set_title(name.split("_", 1)[1].replace("_", " "))
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"stage3a_plot_{name}.png"), dpi=130)
            plt.close(fig)

        # 6. cross-batch alignment, direct vs FBGN
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for m in methods:
            cv = [r["C_V"] for r in rows if r["method"] == m and r["eta_frac"] == 1.0]
            if cv:
                ax.scatter(range(len(cv)), cv, s=22, alpha=0.8, label=m, color=colors[m])
        ax.axhline(0, lw=0.9, color="k")
        ax.set_xlabel("model batch index")
        ax.set_ylabel("C_V = -d_V/(||g_V|| ||p||)   (>0 = descends the trust objective)")
        ax.set_title("cross-batch directional alignment, direct vs FBGN")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "stage3a_plot_6_alignment.png"), dpi=130)
        plt.close(fig)
        print(f"[plots] written to {out_dir}")

    with open(os.path.join(out_dir, "stage3a_table.md"), "w") as f:
        f.write("\n".join(table_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
