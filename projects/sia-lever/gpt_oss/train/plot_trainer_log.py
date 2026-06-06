"""Plot a HF Trainer log (loss / reward / grad_norm vs step) from <out>/trainer_state.json.

Always called at the end of training so an adapter ships with a loss curve even without wandb.
Standalone: python gpt_oss/train/plot_trainer_log.py <adapter_dir>
"""

import json
import os
import sys


def plot_from_dir(out_dir):
    state = os.path.join(out_dir, "trainer_state.json")
    if not os.path.exists(state):
        print(f"[plot_trainer_log] no trainer_state.json in {out_dir} (skipping)")
        return None
    hist = json.load(open(state)).get("log_history", [])
    series = {}
    for e in hist:
        step = e.get("step")
        for k in ("loss", "reward", "rewards/chosen", "grad_norm", "learning_rate"):
            if k in e and step is not None:
                series.setdefault(k, []).append((step, e[k]))
    if not series:
        print("[plot_trainer_log] no plottable scalars")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in ("loss", "reward", "rewards/chosen") if k in series] or list(series)[:2]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        xs = [s for s, _ in series[k]]
        ys = [v for _, v in series[k]]
        ax.plot(xs, ys, marker="o", ms=3)
        ax.set_title(k); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    fig.suptitle(f"training curve — {os.path.basename(out_dir.rstrip('/'))}")
    fig.tight_layout()
    png = os.path.join(out_dir, "training_curve.png")
    fig.savefig(png, dpi=130); plt.close(fig)
    with open(os.path.join(out_dir, "training_curve.json"), "w") as f:
        json.dump({k: v for k, v in series.items()}, f, indent=2)
    print(f"[plot_trainer_log] saved {png}")
    return png


if __name__ == "__main__":
    plot_from_dir(sys.argv[1] if len(sys.argv) > 1 else ".")
