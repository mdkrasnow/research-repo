#!/usr/bin/env python3
"""Plot the CSV/JSON emitted by analyze_training_regression.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text())
    with args.trace.open() as handle:
        rows = list(csv.DictReader(handle))
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    throughput = [float(row["steps_per_sec"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(steps, losses, linewidth=0.7, color="#3366aa")
    axes[0, 0].axvspan(1_516_000, 1_520_000, color="#cc3311", alpha=0.15, label="transition")
    axes[0, 0].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("Training loss")
    axes[0, 0].set_title("Epoch 15→40 loss trace")
    axes[0, 0].legend()

    local = [(step, loss, sps) for step, loss, sps in zip(steps, losses, throughput) if 1_500_000 <= step <= 1_550_000]
    local_steps = [row[0] for row in local]
    local_losses = [row[1] for row in local]
    local_sps = [row[2] for row in local]
    axes[0, 1].plot(local_steps, local_losses, linewidth=0.9, color="#3366aa", label="loss")
    axes[0, 1].axvspan(1_516_000, 1_520_000, color="#cc3311", alpha=0.15)
    axes[0, 1].set_xlabel("Optimizer step")
    axes[0, 1].set_ylabel("Loss", color="#3366aa")
    throughput_axis = axes[0, 1].twinx()
    throughput_axis.plot(local_steps, local_sps, linewidth=0.55, alpha=0.55, color="#228833", label="throughput")
    throughput_axis.set_ylabel("Steps / second", color="#228833")
    axes[0, 1].set_title("1.50M→1.55M event zoom")

    checkpoints = analysis["checkpoints"]
    checkpoint_steps = [row["step"] for row in checkpoints]
    axes[1, 0].plot(checkpoint_steps, [row["ema_mean_loss"] for row in checkpoints], marker="o", color="#3366aa")
    axes[1, 0].set_xlabel("Checkpoint step")
    axes[1, 0].set_ylabel("EMA fixed-bank loss")
    axes[1, 0].set_title("Persistent checkpoint degradation")

    axes[1, 1].plot(checkpoint_steps, [row["ema_mean_cosine"] for row in checkpoints], marker="o", color="#ee7733", label="field cosine")
    axes[1, 1].set_xlabel("Checkpoint step")
    axes[1, 1].set_ylabel("EMA field cosine", color="#ee7733")
    hessian_axis = axes[1, 1].twinx()
    hessian_axis.plot(checkpoint_steps, [row["ema_mean_hessian_vector_norm"] for row in checkpoints], marker="o", color="#228833", label="Hessian-vector norm")
    hessian_axis.set_ylabel("EMA Hessian-vector norm", color="#228833")
    axes[1, 1].set_title("Alignment falls while curvature decreases")

    fig.suptitle("Direct-energy regression: training event and checkpoint consequences")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
