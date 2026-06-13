"""Rung-3 KERNEL: translate weight-retrains into DOLLARS saved.

A weight-retrain (W-call) is the expensive operation: it runs a fine-tune on a GPU. We price
each retrain at (GPU $/hr) * (hours per retrain). Defaults: H200 ~ $4.63/hr, 0.5 hr per retrain
=> ~$2.32 per retrain. Both assumptions are stated on the figure and are the ONLY invented
numbers (the cost model); retrain COUNTS come from the 96-episode CUDA eval.

always-W (paper-style scheduler) retrains on every needed episode -> 72 W-calls over the eval.
SIA-Lever reaches the same correctness target with 48 W-calls. We plot cumulative $ spent for
each, assuming retrains accrue evenly across the 96 episodes, and report the total $ saved.

Rung-3 retrain counts are defined here (not yet in ladder_results.tsv) as the single source for
the rung-3 figures.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUT = os.path.join(PROJ, "results", "regret_dollars.png")

N_EPISODES = 96
RETRAINS_PAPER = 72      # paper-style W+H scheduler
RETRAINS_LEVER = 48      # SIA-Lever (ours)

# --- cost model (the only assumed numbers) ---
GPU_DOLLARS_PER_HR = 4.63   # H200 on-demand, approx
HOURS_PER_RETRAIN = 0.5     # assumed wall-clock per fine-tune retrain
COST_PER_RETRAIN = GPU_DOLLARS_PER_HR * HOURS_PER_RETRAIN  # ~$2.32


def cumulative_dollars(total_retrains):
    """Spread `total_retrains` evenly across episodes; return cumulative $ per episode."""
    ep = np.arange(N_EPISODES + 1)
    retrains = total_retrains * ep / N_EPISODES
    return ep, retrains * COST_PER_RETRAIN


def main():
    ep, paper = cumulative_dollars(RETRAINS_PAPER)
    _, lever = cumulative_dollars(RETRAINS_LEVER)
    saved = paper[-1] - lever[-1]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(ep, paper, lw=3, color="#999999", label=f"always-W / paper-style ({RETRAINS_PAPER} retrains)")
    ax.plot(ep, lever, lw=3, color="#55A868", label=f"SIA-Lever — ours ({RETRAINS_LEVER} retrains)")
    ax.fill_between(ep, lever, paper, color="#55A868", alpha=0.18)

    ax.text(N_EPISODES, paper[-1], f"  ${paper[-1]:.0f}", va="center", fontsize=13,
            fontweight="bold", color="#666666")
    ax.text(N_EPISODES, lever[-1], f"  ${lever[-1]:.0f}", va="center", fontsize=13,
            fontweight="bold", color="#2E7D32")
    ax.annotate(f"${saved:.0f} saved\n({100*(1-RETRAINS_LEVER/RETRAINS_PAPER):.0f}% less retrain spend)",
                xy=(N_EPISODES * 0.72, (paper[int(N_EPISODES*0.72)] + lever[int(N_EPISODES*0.72)]) / 2),
                fontsize=14, fontweight="bold", color="#2E7D32",
                bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#2E7D32"))

    ax.set_xlabel("eval episode", fontsize=15)
    ax.set_ylabel("cumulative retrain cost ($)", fontsize=15)
    ax.set_xlim(0, N_EPISODES)
    ax.set_ylim(0, paper[-1] * 1.15)
    ax.set_title("Rung-3: SIA-Lever cuts GPU retrain spend at the same correctness target",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(alpha=0.25)

    caption = (f"Cost model (assumed): GPU ${GPU_DOLLARS_PER_HR:.2f}/hr (H200) x "
               f"{HOURS_PER_RETRAIN} hr/retrain = ${COST_PER_RETRAIN:.2f}/retrain. "
               f"Retrain counts (72 vs 48) from 96-episode CUDA eval. Retrains assumed even across episodes.")
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9, color="#444444", wrap=True)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}  (saved=${saved:.2f}, cost/retrain=${COST_PER_RETRAIN:.2f})")


if __name__ == "__main__":
    main()
