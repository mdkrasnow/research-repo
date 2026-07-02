"""Rung-2 HARD: the full honest comparison — base, the three LoRA runs, the constant-policy
baselines, and the tiny-floor / latent-ceiling band. Reads results/ladder_results.tsv.

The point of the figure: a bar only counts as a real win if it clears BOTH the base AND the best
constant baseline (always-H = 0.458) without collapsing. Constant baselines are drawn as hatched
bars so a reader can see the degenerate 20ep run sits exactly on always-H.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
TSV = os.path.join(PROJ, "results", "ladder_results.tsv")
OUT = os.path.join(PROJ, "results", "rung2_comparison.png")

# (label, arm-key in tsv, color, hatch)
BARS = [
    ("base", "base_vm", "#4C72B0", None),
    ("LoRA 3-step\n(lr1e-5)", "lora_vm", "#C44E52", None),
    ("LoRA 20ep\n(lr1e-4, imbal)", "lora_20ep", "#8172B2", None),
    ("LoRA balanced-v2\n(lr3e-5)", "lora_balanced_v2", "#55A868", None),
    ("const H", "const_always_H", "#999999", "//"),
    ("const W", "const_always_W", "#bbbbbb", "//"),
    ("const H_THEN_W", "const_always_HTHENW", "#cccccc", "//"),
]


def main():
    acc, floor, ceiling = {}, 0.46, 0.81
    with open(TSV) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["rung"] == "2":
                try:
                    acc[r["arm"]] = float(r["lever_accuracy"])
                except ValueError:
                    pass
                floor = float(r["tiny_floor"]); ceiling = float(r["ceiling"])

    labels = [b[0] for b in BARS if b[1] in acc]
    vals = [acc[b[1]] for b in BARS if b[1] in acc]
    cols = [b[2] for b in BARS if b[1] in acc]
    hatches = [b[3] for b in BARS if b[1] in acc]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between([-0.6, len(vals) - 0.4], floor, ceiling, color="gold", alpha=0.15, zorder=0,
                    label=f"learnable band [{floor:.2f},{ceiling:.2f}]")
    ax.axhline(floor, ls="--", color="darkorange", lw=1.3, label=f"tiny/constant floor {floor:.2f}")
    ax.axhline(ceiling, ls="-", color="green", lw=1.3, label=f"latent ceiling {ceiling:.2f}")
    for i, (v, c, h) in enumerate(zip(vals, cols, hatches)):
        ax.bar(i, v, 0.62, color=c, hatch=h, edgecolor="white")
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("lever accuracy (held-out HARD, 24 ep)")
    ax.set_ylim(0, 0.9)
    ax.set_title("Rung-2 HARD: only balanced-v2 (lr3e-5) clears base AND all constants without collapse")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
