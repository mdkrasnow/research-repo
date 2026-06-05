"""
Phase 1 deliverable (rigor pass). Runs the four-stage lever episode over N seeds, aggregates
mean +/- std + 95% CI, runs a Welch t-test (W-only vs H->W) with Cohen's d on the key structural
metrics, and writes a table + plot.

Stages:
  S1  prediction-only train         (weak harness v0 sees only clean_mse)
  S2  W-only continue training      (more prediction reward)
  S4  H->W structural retrain       (fix verifier, then retrain)

Claim (defensible wording): W-only PRESERVES the shortcut failure (shortcut_sensitivity and
composition_error stay high; the model still solves the broken-symmetry control). H->W REPAIRS it
(shortcut_sensitivity, composition/identity/inverse error -> ~0; neg_control_mse jumps to honest).
"""

import argparse
import copy
import json
import math
import os
import time

import torch
from scipy import stats

from train import train
from verifier import verifier_v1

METRICS = ["clean_mse", "neg_control_mse", "shortcut_sensitivity",
           "composition_error", "identity_error", "inverse_error"]
# Structural metrics where lower = more honest; used for the W-only vs H->W contrast.
CONTRAST = ["composition_error", "shortcut_sensitivity", "identity_error", "inverse_error"]


def one_seed(steps, seed):
    torch.manual_seed(seed)
    m1 = train(steps=steps, objective="prediction_only", seed=seed, log_every=0)
    s1 = verifier_v1(m1, seed=seed)
    m2 = train(model=copy.deepcopy(m1), steps=steps, objective="prediction_only",
               seed=seed + 1, log_every=0)
    s2 = verifier_v1(m2, seed=seed)
    m4 = train(model=copy.deepcopy(m2), steps=steps, objective="structural",
               seed=seed + 2, log_every=0)
    s4 = verifier_v1(m4, seed=seed)
    return {"S1_predonly": s1, "S2_Wonly": s2, "S4_HtoW": s4}


def mean_ci(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    return m, sd, tcrit * se


def cohens_d(a, b):
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (ma - mb) / sp if sp > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

    rows = []
    for s in range(args.seeds):
        print(f"seed {s} ...", flush=True)
        rows.append(one_seed(args.steps, seed=s * 100))

    stages = ["S1_predonly", "S2_Wonly", "S4_HtoW"]
    label = {"S1_predonly": "1 pred-only (v0)", "S2_Wonly": "2 W-only", "S4_HtoW": "4 H->W"}
    cols = {st_: {m: [r[st_][m] for r in rows] for m in METRICS} for st_ in stages}
    table = {st_: {m: mean_ci(cols[st_][m]) for m in METRICS} for st_ in stages}

    verdicts = {st_: {} for st_ in stages}
    for st_ in stages:
        for r in rows:
            v = r[st_]["verdict"]
            verdicts[st_][v] = verdicts[st_].get(v, 0) + 1

    gate_hits = sum(1 for r in rows
                    if r["S4_HtoW"]["neg_control_mse"] > r["S2_Wonly"]["neg_control_mse"])

    # ---- Welch t-test: W-only vs H->W on structural metrics ----
    stat_lines = ["", "## Statistical test — W-only vs H->W (Welch t, two-sided)", "",
                  "| metric | W-only mean | H->W mean | t | p | Cohen's d |",
                  "|---|---|---|---|---|---|"]
    stats_json = {}
    for m in CONTRAST:
        a = cols["S2_Wonly"][m]; b = cols["S4_HtoW"][m]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = cohens_d(a, b)
        stats_json[m] = {"w_mean": sum(a)/len(a), "hw_mean": sum(b)/len(b),
                         "t": float(t), "p": float(p), "cohens_d": d}
        stat_lines.append(f"| {m} | {sum(a)/len(a):.3f} | {sum(b)/len(b):.4f} "
                          f"| {t:.2f} | {p:.2e} | {d:.2f} |")

    # ---- markdown table ----
    lines = [f"# Phase 1 — lever episode ({args.seeds} seeds, {args.steps} steps/stage)", "",
             "Values are mean ± 95% CI. High neg_control_mse = honest (broken-symmetry task has no "
             "real symmetry to exploit). Low neg_control + high composition/identity/inverse error "
             "= shortcut cheating.", "",
             "| Stage | clean_mse | neg_control_mse | shortcut_sens | comp_err | id_err | inv_err | verdicts |",
             "|---|---|---|---|---|---|---|---|"]
    for st_ in stages:
        c = table[st_]
        vd = ", ".join(f"{k}:{n}" for k, n in sorted(verdicts[st_].items()))
        def cell(m): return f"{c[m][0]:.3f}±{c[m][2]:.3f}"
        lines.append(f"| {label[st_]} | {cell('clean_mse')} | {cell('neg_control_mse')} "
                     f"| {cell('shortcut_sensitivity')} | {cell('composition_error')} "
                     f"| {cell('identity_error')} | {cell('inverse_error')} | {vd} |")
    lines += ["", f"**Gate (per-seed S4 neg_control > S2 neg_control): {gate_hits}/{args.seeds} pass**"]
    lines += stat_lines
    lines += ["", "Interpretation: W-only PRESERVES the shortcut (structural errors stay high, "
              "verdict stays shortcut_win); H->W REPAIRS it (all structural errors collapse to ~0, "
              "neg_control becomes honest). The Welch test quantifies the W-only -> H->W gap."]

    md = "\n".join(lines)
    print("\n" + md)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "episode_table.md"), "w") as f:
        f.write(md + "\n")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with open(os.path.join(args.out, f"episode_seeds_{stamp}.json"), "w") as f:
        json.dump({"rows": rows, "table": table, "stats": stats_json, "gate_hits": gate_hits,
                   "seeds": args.seeds, "steps": args.steps}, f, indent=2)

    _plot(table, stages, label, os.path.join(args.out, "episode_plot.png"))
    print(f"\nsaved table + json + plot to {args.out}/")


def _plot(table, stages, label, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xs = list(range(len(stages)))
    xlabels = [label[s] for s in stages]

    for m, color in [("neg_control_mse", "tab:green"), ("shortcut_sensitivity", "tab:red")]:
        means = [table[s][m][0] for s in stages]
        cis = [table[s][m][2] for s in stages]
        axes[0].errorbar(xs, means, yerr=cis, marker="o", capsize=4, label=m, color=color)
    axes[0].set_title("Honesty signals (95% CI)")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(xlabels, rotation=10)
    axes[0].axhline(0, color="gray", lw=0.5); axes[0].legend()

    for m, color in [("composition_error", "tab:blue"), ("identity_error", "tab:purple"),
                     ("inverse_error", "tab:brown"), ("clean_mse", "tab:orange")]:
        means = [table[s][m][0] for s in stages]
        cis = [table[s][m][2] for s in stages]
        axes[1].errorbar(xs, means, yerr=cis, marker="s", capsize=4, label=m, color=color)
    axes[1].set_title("Group-structure / prediction error (95% CI)")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(xlabels, rotation=10)
    axes[1].legend()

    fig.suptitle("SIA lever: W-only preserves the shortcut; H->W learns the real group")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
