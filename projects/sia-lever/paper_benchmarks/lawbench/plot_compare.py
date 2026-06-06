"""Bar chart: our LawBench runs vs paper reference numbers.

Reads any lawbench_eval.json files under adapters/ (top1_accuracy) plus optional --base / --sia-h
accuracies, and plots them next to the paper references (13.5 / 50.0 / 70.1).
Marks reduced-split runs. Saves plots/lawbench_compare.png.
"""

import argparse
import glob
import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = {"paper:init": 13.5, "paper:SIA-H": 50.0, "paper:SIA-W+H": 70.1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=float, default=None, help="base gpt-oss top-1 %")
    ap.add_argument("--sia-h", type=float, default=None, help="our official SIA-H top-1 %")
    args = ap.parse_args()

    bars = dict(PAPER)
    reduced = set()
    if args.base is not None:
        bars["ours:base"] = args.base
    if args.sia_h is not None:
        bars["ours:SIA-H"] = args.sia_h
    for jp in glob.glob(os.path.join(PROJ, "adapters", "**", "lawbench_eval.json"), recursive=True):
        try:
            r = json.load(open(jp))
            acc = r.get("top1_accuracy")
            if acc is None:
                continue
            name = "ours:LoRA-W" + ("*" if r.get("reduced_split") else "")
            bars[name] = acc * 100
            if r.get("reduced_split"):
                reduced.add(name)
        except Exception:
            pass

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(bars)
    vals = [bars[n] for n in names]
    colors = ["tab:gray" if n.startswith("paper") else "tab:purple" for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b = ax.bar(names, vals, color=colors)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("top-1 accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("LawBench: ours vs paper" + ("  (* = reduced split, not comparable)" if reduced else ""))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    os.makedirs(os.path.join(PROJ, "plots"), exist_ok=True)
    out = os.path.join(PROJ, "plots", "lawbench_compare.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")
    if reduced:
        print("NOTE: reduced-split rows marked * — do NOT compare to paper headline.")


if __name__ == "__main__":
    main()
