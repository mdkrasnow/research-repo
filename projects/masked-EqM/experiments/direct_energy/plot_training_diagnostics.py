"""Plot and summarize full-screen EqM training logs.

The full-screen trainer currently logs loss and throughput, but not field-level
diagnostics.  This utility extracts what is available and reports robust local
loss slopes so optimization speed can be compared without relying on a single
minibatch.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOSS_RE = re.compile(r"step=(\d+).*?Train Loss: ([0-9.]+).*?Train Steps/Sec: ([0-9.]+)")


def parse(path: Path) -> dict[str, np.ndarray]:
    text = path.read_text(errors="replace")
    rows = LOSS_RE.findall(text)
    if not rows:
        raise ValueError(f"no training rows found in {path}")
    arr = np.asarray(rows, dtype=float)
    return {"step": arr[:, 0], "loss": arr[:, 1], "steps_per_sec": arr[:, 2]}


def rolling_mean(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(width, len(x)))
    return np.convolve(x, np.ones(width) / width, mode="valid")


def robust_slope(step: np.ndarray, loss: np.ndarray, fraction: float = 0.2) -> float:
    n = max(5, int(len(step) * fraction))
    x = step[-n:]
    y = loss[-n:]
    return float(np.polyfit(x, y, 1)[0])


def rolling_slope(step: np.ndarray, loss: np.ndarray, width: int) -> np.ndarray:
    """Fit a local slope, avoiding finite-difference artifacts at duplicate steps."""
    width = max(5, min(width, len(loss)))
    out = []
    for i in range(width - 1, len(loss)):
        x = step[i - width + 1 : i + 1]
        y = loss[i - width + 1 : i + 1]
        if np.ptp(x) == 0:
            out.append(np.nan)
        else:
            out.append(np.polyfit(x, y, 1)[0])
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs: dict[str, list[dict[str, np.ndarray]]] = {"none": [], "dot": [], "direct": []}
    for arm in runs:
        for path in sorted((args.logs).glob(f"{arm}_seed*/log.txt")):
            runs[arm].append(parse(path))

    summary: dict[str, dict[str, float | list[float]]] = {}
    for arm, items in runs.items():
        slopes = [robust_slope(x["step"], x["loss"]) for x in items]
        first = [float(np.mean(x["loss"][: max(1, len(x["loss"]) // 10)])) for x in items]
        final = [float(np.mean(x["loss"][-max(1, len(x["loss"]) // 10):])) for x in items]
        summary[arm] = {
            "seeds": len(items),
            "first_tenth_loss_mean": float(np.mean(first)),
            "final_tenth_loss_mean": float(np.mean(final)),
            "final_tenth_loss_std": float(np.std(final, ddof=1)) if len(final) > 1 else 0.0,
            "final_slope_loss_per_step_mean": float(np.mean(slopes)),
            "final_slope_loss_per_step_by_seed": slopes,
            "steps_per_sec_by_seed": [float(np.mean(x["steps_per_sec"][-10:])) for x in items],
        }

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    colors = {"none": "#555555", "dot": "#1f77b4", "direct": "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for arm, items in runs.items():
        for i, x in enumerate(items):
            ax.plot(x["step"], x["loss"], alpha=0.18, color=colors[arm])
        n = min(len(x["loss"]) for x in items)
        mat = np.stack([x["loss"][:n] for x in items])
        ax.plot(items[0]["step"][:n], mat.mean(0), color=colors[arm], lw=2, label=arm)
    ax.set(title="Training loss (thin=seed, thick=mean)", xlabel="optimizer step", ylabel="loss")
    ax.legend()

    ax = axes[0, 1]
    for arm, items in runs.items():
        n = min(len(x["loss"]) for x in items)
        mat = np.stack([rolling_mean(x["loss"][:n], min(25, n)) for x in items])
        steps = items[0]["step"][min(25, n) - 1 : n]
        slopes = np.stack([rolling_slope(x["step"][:n], x["loss"][:n], min(25, n)) for x in items])
        ax.plot(steps, np.nanmean(slopes, 0), color=colors[arm], lw=2, label=arm)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Smoothed loss slope", xlabel="optimizer step", ylabel="d(loss)/d(step)")
    ax.legend()

    ax = axes[1, 0]
    for arm, items in runs.items():
        vals = [robust_slope(x["step"], x["loss"]) for x in items]
        ax.scatter([arm] * len(vals), vals, color=colors[arm], s=65)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Final 20% loss slope by seed", ylabel="loss / optimizer step")

    ax = axes[1, 1]
    for arm, items in runs.items():
        vals = [np.mean(x["steps_per_sec"][-10:]) for x in items]
        ax.scatter([arm] * len(vals), vals, color=colors[arm], s=65)
    ax.set(title="Steady-state throughput", ylabel="steps/sec")

    fig.suptitle("EqM full-screen training dynamics")
    fig.savefig(args.out / "training_dynamics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
