"""Plot the SIA-Lever ladder results from results/ladder_results.tsv.

Two figures:
  1. ladder_bracket.png — per rung: base accuracy vs tiny floor vs latent ceiling (the band a
     learned selector must climb). LoRA bars drawn when a measured lora_accuracy exists.
  2. per_mode_base_hard.png — rung-2 base per-mode accuracy (where base fails: compound faults).

Pure-stdlib parse + matplotlib. Re-run any time; reads NA gracefully.
Usage: python3 scripts/plot_ladder.py
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
TSV = os.path.join(PROJ, "results", "ladder_results.tsv")
OUT = os.path.join(PROJ, "results")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load():
    with open(TSV) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def plot_bracket(rows):
    # one group per rung; base bar = first arm starting "base", lora bar = last measured arm
    # starting "lora" (so a retrain overrides an earlier negative result). floor/ceiling shaded.
    by_rung = {}
    for r in rows:
        acc = _f(r["lever_accuracy"])
        g = by_rung.setdefault(r["rung"], {"label": f"R{r['rung']}\n{r['task']}",
                                           "base": None, "floor": _f(r["tiny_floor"]),
                                           "ceiling": _f(r["ceiling"]), "lora": None})
        if r["arm"].startswith("base") and acc is not None and g["base"] is None:
            g["base"] = acc
        if r["arm"].startswith("lora") and acc is not None:
            g["lora"] = acc  # last wins
    rungs = [by_rung[k] for k in sorted(by_rung)]

    n = len(rungs)
    fig, ax = plt.subplots(figsize=(2.6 * n + 2, 5))
    x = range(n)
    w = 0.34
    for i, g in enumerate(rungs):
        # band: floor..ceiling shaded
        if g["floor"] is not None and g["ceiling"] is not None:
            ax.fill_between([i - 0.45, i + 0.45], g["floor"], g["ceiling"],
                            color="gold", alpha=0.18, zorder=0)
            ax.hlines(g["floor"], i - 0.45, i + 0.45, color="darkorange", lw=1.5,
                      label="tiny floor" if i == 0 else None)
            ax.hlines(g["ceiling"], i - 0.45, i + 0.45, color="green", lw=1.5,
                      label="latent ceiling" if i == 0 else None)
        if g["base"] is not None:
            ax.bar(i - w / 2, g["base"], w, color="#4C72B0",
                   label="base gpt-oss-120b" if i == 0 else None)
            ax.text(i - w / 2, g["base"] + 0.02, f"{g['base']:.2f}", ha="center", fontsize=9)
        if g["lora"] is not None:
            ax.bar(i + w / 2, g["lora"], w, color="#C44E52",
                   label="LoRA" if i == 0 else None)
            ax.text(i + w / 2, g["lora"] + 0.02, f"{g['lora']:.2f}", ha="center", fontsize=9)
    ax.axhline(0.333, ls=":", color="grey", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([g["label"] for g in rungs])
    ax.set_ylabel("lever accuracy (held-out, measured)")
    ax.set_ylim(0, 1.05)
    ax.set_title("SIA-Lever ladder: base vs floor→ceiling band per rung")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "ladder_bracket.png")
    fig.savefig(p, dpi=130)
    print(f"saved {p}")


def plot_per_mode():
    # rung-2 base per-mode from the eval json if present, else skip
    ev = os.path.join(OUT, "gpt_oss", "base_hard_eval.json")
    if not os.path.exists(ev):
        print("no base_hard_eval.json — skip per-mode plot")
        return
    pm = json.load(open(ev)).get("per_mode_accuracy", {})
    if not pm:
        return
    modes = list(pm.keys())
    vals = [pm[m] for m in modes]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(modes)), 4.5))
    ax.bar(range(len(modes)), vals, color="#4C72B0")
    ax.axhline(0.46, ls="--", color="darkorange", label="tiny floor 0.46")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Rung-2 base gpt-oss-120b: per-mode lever accuracy (fails on compound faults)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "per_mode_base_hard.png")
    fig.savefig(p, dpi=130)
    print(f"saved {p}")


if __name__ == "__main__":
    plot_bracket(load())
    plot_per_mode()
