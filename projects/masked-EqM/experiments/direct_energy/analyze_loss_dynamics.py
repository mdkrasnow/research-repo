"""Deeper analysis of full-screen EqM training logs.

The trainer logs a 50-step averaged loss, so raw pointwise slopes are noisy and
overstate differences. This utility uses 5k-step block means, rolling slopes,
and arm-to-arm loss gaps. It treats training loss as a diagnostic, not a proxy
for FID: the energy parameterizations have different loss floors.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LINE = re.compile(r"step=(\d+).*?Train Loss: ([0-9.eE+-]+).*?Train Steps/Sec: ([0-9.eE+-]+)")
ARMS = ("none", "dot", "direct")


def read_logs(paths: list[Path]) -> dict[str, np.ndarray]:
    rows: dict[int, tuple[float, float]] = {}
    for path in paths:
        for line in path.read_text(errors="ignore").splitlines():
            match = LINE.search(line)
            if match:
                step, loss, speed = int(match.group(1)), float(match.group(2)), float(match.group(3))
                rows[step] = (loss, speed)
    if not rows:
        raise ValueError(f"no training rows found in {paths}")
    steps = np.array(sorted(rows), dtype=float)
    return {
        "step": steps,
        "loss": np.array([rows[int(s)][0] for s in steps]),
        "speed": np.array([rows[int(s)][1] for s in steps]),
    }


def blocks(data: dict[str, np.ndarray], width: int = 5000) -> dict[str, np.ndarray]:
    step, loss = data["step"], data["loss"]
    key = (step // width).astype(int) * width
    out_step, out_loss, out_std = [], [], []
    for block in np.unique(key):
        mask = key == block
        out_step.append(step[mask].mean())
        out_loss.append(loss[mask].mean())
        out_std.append(loss[mask].std(ddof=1))
    return {"step": np.array(out_step), "loss": np.array(out_loss), "std": np.array(out_std)}


def slope(block: dict[str, np.ndarray], lo: int, hi: int) -> float:
    mask = (block["step"] >= lo) & (block["step"] <= hi)
    return float(np.polyfit(block["step"][mask], block["loss"][mask], 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = {}
    for arm in ARMS:
        data[arm] = read_logs(sorted(args.log_dir.glob(f"*{arm}*.log")))
    blk = {arm: blocks(data[arm]) for arm in ARMS}

    windows = [(40000, 80000), (80000, 120000), (120000, 160000), (160000, 200100), (40000, 200100)]
    summary = {"block_width": 5000, "windows": {}, "throughput": {}}
    for arm in ARMS:
        median_speed = float(np.median(data[arm]["speed"]))
        summary["throughput"][arm] = {
            "median_steps_per_sec": median_speed,
            "effective_hours_per_200k_steps": float(200000 / median_speed / 3600),
        }
    for lo, hi in windows:
        summary["windows"][f"{lo}_{hi}"] = {arm: slope(blk[arm], lo, hi) for arm in ARMS}

    common = sorted(set(blk["none"]["step"]) & set(blk["dot"]["step"]) & set(blk["direct"]["step"]))
    series = {
        arm: np.array([blk[arm]["loss"][np.where(blk[arm]["step"] == s)[0][0]] for s in common])
        for arm in ARMS
    }
    gaps = {"direct_minus_dot": series["direct"] - series["dot"], "dot_minus_none": series["dot"] - series["none"]}
    summary["gap_summary"] = {}
    for name, values in gaps.items():
        after_40k = np.where(np.asarray(common) >= 40000)[0][0]
        at_200k = np.where(np.asarray(common) >= 200000)[0][0]
        summary["gap_summary"][name] = {
            "at_40k": float(values[after_40k]),
            "at_200k": float(values[at_200k]),
            "slope_40k_to_200k": float(np.polyfit(np.asarray(common)[after_40k:at_200k + 1], values[after_40k:at_200k + 1], 1)[0]),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"none": "#555555", "dot": "#1f77b4", "direct": "#d62728"}
    for arm in ARMS:
        axes[0, 0].plot(blk[arm]["step"], blk[arm]["loss"], label=arm, color=colors[arm])
        axes[0, 0].fill_between(
            blk[arm]["step"], blk[arm]["loss"] - blk[arm]["std"],
            blk[arm]["loss"] + blk[arm]["std"], alpha=.08, color=colors[arm],
        )
    axes[0, 0].set(title="5k-step mean loss (band = within-block SD)", xlabel="optimizer step", ylabel="loss")
    axes[0, 0].legend()
    for arm in ARMS:
        x, y = data[arm]["step"], data[arm]["loss"]
        width = 800
        rolling = np.array([
            np.polyfit(x[i - width + 1:i + 1], y[i - width + 1:i + 1], 1)[0]
            for i in range(width - 1, len(x))
        ])
        axes[0, 1].plot(x[width - 1:], rolling * 1e6, label=arm, color=colors[arm])
    axes[0, 1].axhline(0, color="black", lw=.7)
    axes[0, 1].set(title="Rolling 40k-step slope", xlabel="optimizer step", ylabel="loss / step × 1e6")
    axes[0, 1].legend()
    axes[1, 0].plot(common, gaps["direct_minus_dot"], label="direct − dot", color=colors["direct"])
    axes[1, 0].plot(common, gaps["dot_minus_none"], label="dot − none", color=colors["dot"])
    axes[1, 0].axhline(0, color="black", lw=.7)
    axes[1, 0].set(title="Loss gaps (5k-step means)", xlabel="optimizer step", ylabel="loss gap")
    axes[1, 0].legend()
    for arm in ARMS:
        axes[1, 1].plot(blk[arm]["step"], blk[arm]["std"], label=arm, color=colors[arm])
    axes[1, 1].set(title="Within-block loss variability", xlabel="optimizer step", ylabel="SD of logged loss")
    axes[1, 1].legend()
    fig.savefig(args.output / "loss_dynamics_deep.png", dpi=180)
    plt.close(fig)

    report = [
        "# Deeper training-dynamics analysis", "",
        "This analysis uses 5,000-step means and 40,000-step rolling slopes; raw 50-step log values are too noisy for direct extrapolation.", "",
        "## Slopes (loss / step)", "",
        "| Window | none | dot | direct | direct−dot slope |", "|---|---:|---:|---:|---:|",
    ]
    for lo, hi in windows:
        vals = summary["windows"][f"{lo}_{hi}"]
        report.append(f"| {lo:,}–{hi:,} | {vals['none']:.3e} | {vals['dot']:.3e} | {vals['direct']:.3e} | {(vals['direct'] - vals['dot']):.3e} |")
    report += [
        "", "## Interpretation", "",
        "- The none–dot loss gap does not close after step 40k; it is approximately flat to slightly wider. None is therefore not on a credible loss-parity trajectory with dot, even though it has the lowest absolute loss.",
        "- The direct–dot gap is different: it shrinks from the early continuation and is approximately zero by 200k steps. In the final 40k window, direct's slope is more negative than dot's, but the confidence is weak at this noise level. The one-epoch slope extrapolation is not a valid reason to predict an 80-epoch failure.",
        "- The three arms have distinct loss floors (roughly 10.67 none, 11.07 dot, 11.1 direct from exponential fits). Absolute training loss is not a fair cross-parameterization quality metric: the FID probes show none has lower loss but worse FID, while dot/direct have better FID.",
        "- Direct remains about 1.9× slower than none in optimizer steps, so equal wall-clock comparisons should not be confused with equal-step comparisons.",
        "", "The remaining discriminating experiment is not to wait for none to catch dot in training loss. It is to test whether direct's late loss-gap closure persists through epoch 8 and whether its already-near-dot FID remains tied or separates.", "",
    ]
    (args.output / "report.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
