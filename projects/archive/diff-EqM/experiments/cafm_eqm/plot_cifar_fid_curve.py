"""Plot v10 CIFAR FID trajectory + mining diagnostics from train_log.tsv.

Produces paper Figure 2: FID curve + mining ratio overlay across 150 epochs
of v10 CIFAR sanity training (job 13868114).

Usage (local, requires matplotlib + numpy + the persisted train_log.tsv):
  python plot_cifar_fid_curve.py \
    --log projects/diff-EqM/results/variant_v10_hard_example_13868114_seed0/train_log.tsv \
    --out projects/diff-EqM/results/v10_cifar_fid_curve.pdf

If train_log.tsv has not been synced from cluster yet, run:
  bash scripts/cluster/ssh.sh "cat /n/home03/mkrasnow/research-repo/projects/diff-EqM/results/variant_v10_hard_example_13868114_seed0/train_log.tsv" > /tmp/v10_log.tsv
  python plot_cifar_fid_curve.py --log /tmp/v10_log.tsv --out v10_cifar_fid_curve.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse(log_path: Path):
    """Return (epoch_rows, eval_rows, final). Row format from _common.train_loop:
      epoch\tstep\ttotal\tbase\thard\tratio\tdelta_norm\telapsed
      eval\t<epoch>\t<fid>
      final\t<num_samples>\t<fid>
    """
    epoch_rows = []
    eval_rows = []
    final = None
    for line in log_path.read_text(errors="replace").splitlines():
        parts = line.strip().split("\t")
        if not parts:
            continue
        if parts[0] == "eval" and len(parts) >= 3:
            eval_rows.append({"epoch": int(parts[1]), "fid": float(parts[2])})
        elif parts[0] == "final" and len(parts) >= 3:
            final = {"num_samples": int(parts[1]), "fid": float(parts[2])}
        elif len(parts) >= 7:
            try:
                epoch_rows.append({
                    "epoch": int(parts[0]),
                    "step": int(parts[1]),
                    "total": float(parts[2]),
                    "base": float(parts[3]),
                    "hard": float(parts[4]),
                    "ratio": float(parts[5]),
                    "delta_norm": float(parts[6]),
                })
            except ValueError:
                continue
    return epoch_rows, eval_rows, final


def make_plot(epoch_rows, eval_rows, final, out_path: Path):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        raise SystemExit("matplotlib required; pip install matplotlib")

    fig, (ax_loss, ax_fid, ax_ratio) = plt.subplots(
        3, 1, figsize=(7, 9), sharex=True,
        gridspec_kw={"hspace": 0.18},
    )

    epochs = [r["epoch"] for r in epoch_rows]

    # Top: base + hard loss
    ax_loss.plot(epochs, [r["base"] for r in epoch_rows], label=r"$L_\mathrm{EqM}$",
                 color="#1f77b4", linewidth=1.4)
    ax_loss.plot(epochs, [r["hard"] for r in epoch_rows], label=r"$L_\mathrm{v10}$",
                 color="#d62728", linewidth=1.4, alpha=0.85)
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right", frameon=False)
    ax_loss.grid(alpha=0.25)

    # Middle: FID (eval + final)
    if eval_rows:
        ax_fid.plot(
            [r["epoch"] for r in eval_rows],
            [r["fid"] for r in eval_rows],
            "o-", color="#2ca02c", label="FID @1K (during training)",
            linewidth=1.4, markersize=6,
        )
    if final:
        ax_fid.axhline(final["fid"], color="#2ca02c", linestyle="--", alpha=0.5)
        ax_fid.annotate(
            f"final FID@{final['num_samples']}: {final['fid']:.2f}",
            xy=(epochs[-1], final["fid"]), xytext=(-110, 8),
            textcoords="offset points", color="#2ca02c", fontsize=9,
        )
    # Vanilla reference line
    ax_fid.axhline(14.17, color="#7f7f7f", linestyle=":", alpha=0.7,
                   label="vanilla v00 R4 (14.17)")
    ax_fid.set_ylabel("FID")
    ax_fid.set_yscale("log")
    ax_fid.legend(loc="upper right", frameon=False, fontsize=9)
    ax_fid.grid(alpha=0.25, which="both")

    # Bottom: mining ratio + delta_norm
    ax_ratio.plot(epochs, [r["ratio"] for r in epoch_rows],
                  color="#9467bd", linewidth=1.4, label=r"$L_\mathrm{v10}/L_\mathrm{EqM}$")
    ax_ratio.axhline(1.0, color="black", linestyle="--", alpha=0.4)
    ax_ratio.set_ylabel("Ratio", color="#9467bd")
    ax_ratio.tick_params(axis="y", labelcolor="#9467bd")
    ax_ratio.set_xlabel("Epoch")
    ax_ratio.grid(alpha=0.25)

    ax_ratio2 = ax_ratio.twinx()
    ax_ratio2.plot(epochs, [r["delta_norm"] for r in epoch_rows],
                   color="#ff7f0e", linewidth=1.0, alpha=0.6,
                   label=r"$\|\delta\|$")
    ax_ratio2.set_ylabel(r"$\|\delta\|$", color="#ff7f0e")
    ax_ratio2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_ratio2.set_ylim(0.0, 0.5)

    fig.suptitle("v10 PGD hard-example mining on EqM-CIFAR (job 13868114)", fontsize=12)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Saved plot to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not args.log.exists():
        raise SystemExit(f"log not found: {args.log}")

    epoch_rows, eval_rows, final = parse(args.log)
    print(f"Parsed {len(epoch_rows)} epoch rows, {len(eval_rows)} eval rows, "
          f"final FID = {final['fid'] if final else 'N/A'}")
    make_plot(epoch_rows, eval_rows, final, args.out)


if __name__ == "__main__":
    main()
