"""Aggregate checkpoint-forensics JSON into plots and a concise report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(args):
    root = Path(args.input)
    runs = []
    for path in sorted(root.glob("direct_*.json")):
        payload = json.loads(path.read_text())
        step = int(payload["checkpoint_label"].split("_")[-1])
        runs.append((step, payload))

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    for weight, style in (("model", "--"), ("ema", "-")):
        for step, payload in runs:
            rows = payload["weights"][weight]["field_by_t"]
            label = f"{step / 1e6:.2f}M {weight}"
            axes[0, 0].plot([r["t"] for r in rows], [r["loss_mean"] for r in rows], style, label=label)
            axes[0, 1].plot([r["t"] for r in rows], [r["cosine_mean"] for r in rows], style, label=label)
            axes[0, 2].plot([r["t"] for r in rows], [r["norm_ratio_mean"] for r in rows], style, label=label)
            curvature = payload["weights"][weight]["curvature"]
            axes[1, 0].plot([r["t"] for r in curvature],
                            [r["hessian_vector_norm_mean"] for r in curvature], style, label=label)
            trajectory = payload["weights"][weight]["sampling"]["rows"]
            axes[1, 1].plot([r["step"] for r in trajectory],
                            [r["field_norm_mean"] for r in trajectory], style, label=label)
            axes[1, 2].plot([r["step"] for r in trajectory],
                            [r["fixed_t_energy_delta_mean"] for r in trajectory], style, label=label)
    titles = ["Field loss by t", "Field-target cosine by t", "Field/target norm ratio by t",
              "Directional Hessian norm", "Sampling field norm", "Fixed-t energy change per GD step"]
    ylabels = ["MSE", "cosine", "norm ratio", "||H v||", "||field||", "E(x+eta f,t)-E(x,t)"]
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set_title(title); axis.set_xlabel("t" if "by t" in title or "Hessian" in title else "sampling step")
        axis.set_ylabel(ylabel); axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(root / "checkpoint_forensics.png", dpi=180)
    plt.close(figure)

    state = json.loads((root / "state_deltas.json").read_text())
    intervals = [f"{row['left_step']//1000}k-{row['right_step']//1000}k" for row in state]
    groups = ["energy_head", "t_embedder", "x_embedder", "blocks.0", "blocks.1", "blocks.11"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(intervals)); width = 0.12
    for index, group in enumerate(groups):
        relative = [row["groups"][group]["model_delta_norm"] /
                    max(row["groups"][group]["model_left_norm"], 1e-12) for row in state]
        adam = [row["groups"][group].get("adam_exp_avg_delta_norm", 0.0) for row in state]
        axes[0].bar(x + (index - 2.5) * width, relative, width, label=group)
        axes[1].bar(x + (index - 2.5) * width, adam, width, label=group)
    axes[0].set(title="Relative raw-parameter displacement", ylabel="||delta theta|| / ||theta||")
    axes[1].set(title="Adam first-moment displacement", ylabel="||delta exp_avg||")
    for axis in axes:
        axis.set_xticks(x, intervals); axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(root / "checkpoint_state_deltas.png", dpi=180); plt.close(fig)

    summaries = {}
    for step, payload in runs:
        summaries[str(step)] = {}
        for weight in ("model", "ema"):
            fields = payload["weights"][weight]["field_by_t"]
            curvature = payload["weights"][weight]["curvature"]
            trajectory = payload["weights"][weight]["sampling"]["rows"]
            summaries[str(step)][weight] = {
                "mean_loss": float(np.mean([r["loss_mean"] for r in fields])),
                "mean_cosine": float(np.mean([r["cosine_mean"] for r in fields])),
                "mean_norm_ratio": float(np.mean([r["norm_ratio_mean"] for r in fields])),
                "mean_hessian_vector_norm": float(np.mean([r["hessian_vector_norm_mean"] for r in curvature])),
                "fixed_t_energy_increase_fraction": float(np.mean(
                    [r["fixed_t_energy_increase_fraction"] for r in trajectory])),
            }
    (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    fid_paths = sorted((root / "fid500").glob("direct_*.json"))
    if fid_paths:
        fid_rows = [(int(path.stem.split("_")[-1]), json.loads(path.read_text())["fid"])
                    for path in fid_paths]
        fid_rows.sort()
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot([step / 1e6 for step, _ in fid_rows], [fid for _, fid in fid_rows],
                  marker="o", linewidth=2)
        axis.axvline(1.51755, color="red", linestyle="--", alpha=0.7,
                     label="observed optimizer shock")
        axis.set(title="Matched 500-sample FID around direct optimizer shock",
                 xlabel="optimizer step (millions)", ylabel="FID-500")
        axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
        fig.savefig(root / "fid500_localization.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    main(parser.parse_args())
