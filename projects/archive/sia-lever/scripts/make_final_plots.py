#!/usr/bin/env python3
"""Regenerate plots/final_comparison.png from results/final_comparison.csv (no reruns)."""

import csv
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    csv_path = os.path.join(PROJ, "results", "final_comparison.csv")
    if not os.path.exists(csv_path):
        raise SystemExit("run scripts/make_final_table.py first (no final_comparison.csv)")
    rows = list(csv.DictReader(open(csv_path)))
    names = [r["policy"] for r in rows]
    regret = [float(r["mean_regret"]) for r in rows]
    acc = [float(r["lever_accuracy"]) for r in rows]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].barh(names, regret, color="tab:red"); ax[0].set_title("Mean regret (lower=better)")
    ax[0].invert_yaxis()
    ax[1].barh(names, acc, color="tab:green"); ax[1].set_title("Lever accuracy (higher=better)")
    ax[1].set_xlim(0, 1); ax[1].invert_yaxis()
    fig.suptitle("SIA-Lever-120B: lever-attribution policy comparison (measured regret)")
    fig.tight_layout()
    out = os.path.join(PROJ, "plots", "final_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
