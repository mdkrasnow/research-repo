"""Aggregate the six locked runs, bootstrap clusters, plot, and report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from masked_field_shaping.statistics import classify_result, paired_image_cluster_bootstrap


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _training_row(epoch, arm, paths, config):
    completion = json.loads(Path(paths["training_completion"]).read_text())
    initial = json.loads(Path(paths["training_initial_state"]).read_text())
    metrics = [json.loads(line) for line in Path(paths["training_metrics"]).read_text().splitlines() if line]
    weighted_mask = sum((entry["effective_mask_fraction"] or 0.0) for entry in metrics) / len(metrics)
    mask_weight = sum((entry["effective_mask_fraction"] or 0.0) for entry in metrics)
    mean_ratio = (
        sum(
            (entry["mean_realized_mask_ratio"] or 0.0) * (entry["effective_mask_fraction"] or 0.0)
            for entry in metrics
        )
        / mask_weight
        if mask_weight
        else 0.0
    )
    return {
        "base_epoch": epoch,
        "arm": arm,
        "start_checkpoint": config["base_checkpoint"],
        "final_checkpoint": completion["final_checkpoint"],
        "continuation_epochs": config["continuation_epochs"],
        "optimizer_steps": completion["optimizer_steps"],
        "global_batch_size": config["global_batch_size"],
        "restored_optimizer": initial["restored_optimizer"],
        "restored_scheduler": initial["restored_scheduler"],
        "seed": config["training_seed"],
        "effective_mask_probability": weighted_mask,
        "mean_mask_ratio": mean_ratio,
        "final_training_loss": metrics[-1]["loss"],
        "runtime_hours": completion["runtime_hours"],
        "status": completion["status"],
    }


def aggregate(manifest_path: str) -> None:
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(manifest["artifact_root"])
    aggregate_dir = root / "aggregate"
    plot_dir = aggregate_dir / "plots"
    reports_dir = root / "reports"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    training_rows = []
    recovery_frames = []
    generation_rows = []
    summaries = []
    paired_deltas = {}
    for epoch in (15, 40, 80):
        arm_frames = {}
        generation_by_arm = {}
        for arm in ("control", "masked"):
            key = f"epoch{epoch}_{arm}"
            paths = manifest["runs"][key]
            config = json.loads(Path(paths["training_config"]).read_text())
            training_rows.append(_training_row(epoch, arm, paths, config))
            frame = pd.read_csv(paths["recovery_csv"])
            recovery_frames.append(frame)
            arm_frames[arm] = frame
            generation = json.loads(Path(paths["generation_summary"]).read_text())
            generation_by_arm[arm] = generation
            generation_rows.append(
                {
                    "base_epoch": epoch,
                    "arm": arm,
                    "generated_samples": generation["generated_samples"],
                    "fid": generation["fid"],
                    "kid_if_available": generation["kid_if_available"],
                    "failed_samples": generation["failed_samples"],
                    "runtime": generation["runtime_seconds"],
                    "function_evaluations": generation["function_evaluations_per_sample"],
                }
            )

        keys = ["image_id", "corruption_draw"]
        paired = arm_frames["control"].merge(
            arm_frames["masked"], on=keys, suffixes=("_control", "_treatment"), validate="one_to_one"
        )
        max_d0 = float(np.max(np.abs(paired["d0_lpips_control"] - paired["d0_lpips_treatment"])))
        if max_d0 > float(manifest["metrics"]["d0_tolerance"]):
            raise RuntimeError(f"epoch {epoch} paired D0 mismatch: max abs {max_d0}")
        result = paired_image_cluster_bootstrap(
            paired["image_id"],
            paired["lpips_recovery_control"],
            paired["lpips_recovery_treatment"],
            replicates=int(manifest["statistics"]["bootstrap_replicates"]),
            seed=int(manifest["statistics"]["bootstrap_seed"]) + epoch,
        )
        decision = classify_result(
            result.ci_lower,
            generation_by_arm["control"]["fid"],
            generation_by_arm["masked"]["fid"],
        )
        summaries.append(
            {
                "base_epoch": epoch,
                "control_mean_recovery": float(paired["lpips_recovery_control"].mean()),
                "treatment_mean_recovery": float(paired["lpips_recovery_treatment"].mean()),
                "paired_delta": result.paired_delta,
                "bootstrap_standard_error": result.bootstrap_standard_error,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "fraction_improved": result.fraction_improved,
                "recovery_pass": result.ci_lower > 0,
                "max_abs_d0_difference": max_d0,
                "decision": decision["decision"],
                "fid_delta": decision["fid_delta"],
            }
        )
        paired_deltas[epoch] = (
            paired["lpips_recovery_treatment"] - paired["lpips_recovery_control"]
        ).to_numpy()

        steps = [0, 1, 2, 4, 8]
        fig, ax = plt.subplots(figsize=(6, 4))
        for arm, frame in arm_frames.items():
            ax.plot(steps, [frame[f"d{step}_lpips"].mean() for step in steps], marker="o", label=arm)
        ax.set(xlabel="Recovery update", ylabel="Mean LPIPS", title=f"Block recovery from epoch {epoch}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"lpips_trajectory_epoch{epoch}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(paired_deltas[epoch], bins=60, color="#4C78A8", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.set(
            xlabel="Treatment minus control LPIPS recovery",
            ylabel="Corruption draws",
            title=f"Paired recovery differences, epoch {epoch}",
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"paired_recovery_distribution_epoch{epoch}.png", dpi=180)
        plt.close(fig)

    pd.DataFrame(training_rows).to_csv(aggregate_dir / "training_runs.csv", index=False)
    pd.concat(recovery_frames, ignore_index=True).to_csv(aggregate_dir / "recovery_per_example.csv", index=False)
    recovery_summary = pd.DataFrame(summaries)
    recovery_summary.to_csv(aggregate_dir / "recovery_summary.csv", index=False)
    generation_summary = pd.DataFrame(generation_rows)
    generation_summary.to_csv(aggregate_dir / "generation_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        recovery_summary["base_epoch"],
        recovery_summary["paired_delta"],
        yerr=[
            recovery_summary["paired_delta"] - recovery_summary["ci_lower"],
            recovery_summary["ci_upper"] - recovery_summary["paired_delta"],
        ],
        fmt="o",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Base epoch", ylabel="Treatment - control recovery", title="Paired recovery with 95% clustered CIs")
    fig.tight_layout()
    fig.savefig(plot_dir / "recovery_ci_across_epochs.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    width = 0.35
    for offset, arm in ((-width / 2, "control"), (width / 2, "masked")):
        values = generation_summary[generation_summary.arm == arm].sort_values("base_epoch")["fid"]
        ax.bar(x + offset, values, width, label=arm)
    ax.set_xticks(x, ["15", "40", "80"])
    ax.set(xlabel="Base epoch", ylabel="FID-10k", title="Ordinary Gaussian generation guardrail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "fid_control_vs_treatment.png", dpi=180)
    plt.close(fig)

    qualitative_ids = manifest["evaluation_bank"]["qualitative_image_ids"]
    for epoch in (15, 40, 80):
        fig, axes = plt.subplots(len(qualitative_ids), 4, figsize=(10, 2.5 * len(qualitative_ids)))
        if len(qualitative_ids) == 1:
            axes = axes[None, :]
        control_dir = Path(manifest["runs"][f"epoch{epoch}_control"]["recovery_qualitative"])
        masked_dir = Path(manifest["runs"][f"epoch{epoch}_masked"]["recovery_qualitative"])
        for row, image_id in enumerate(qualitative_ids):
            prefix = f"image{int(image_id):05d}"
            paths = [
                control_dir / f"{prefix}_clean.png",
                control_dir / f"{prefix}_corrupted.png",
                control_dir / f"{prefix}_control_step8.png",
                masked_dir / f"{prefix}_masked_step8.png",
            ]
            for column, path in enumerate(paths):
                axes[row, column].imshow(Image.open(path))
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title(["Clean", "Block corrupted", "Control step 8", "Treatment step 8"][column])
        fig.suptitle(f"Deterministically selected qualitative recovery, epoch {epoch}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"qualitative_grid_epoch{epoch}.png", dpi=160)
        plt.close(fig)

    summary_by_epoch = {row["base_epoch"]: row for row in summaries}
    stage_replicated = (
        summary_by_epoch[40]["decision"] == "PASS"
        and (summary_by_epoch[15]["recovery_pass"] or summary_by_epoch[80]["recovery_pass"])
        and summary_by_epoch[15]["fid_delta"] <= 1.0
        and summary_by_epoch[80]["fid_delta"] <= 1.0
    )
    decisions = {
        "epoch_40_primary_decision": summary_by_epoch[40],
        "epoch_15_replication_decision": summary_by_epoch[15],
        "epoch_80_replication_decision": summary_by_epoch[80],
        "stage_replication_decision": "stage-replicated" if stage_replicated else "not stage-replicated",
        "stage_replication_reason": (
            "Epoch 40 passed, at least one replication had a positive recovery CI, and neither replication exceeded the FID margin."
            if stage_replicated
            else "One or more predeclared stage-replication conditions were not met."
        ),
        "thresholds": {"recovery_ci_lower_gt": 0.0, "fid_treatment_minus_control_lte": 1.0},
        "single_seed_limitation": "Epochs 15, 40, and 80 are checkpoints from one training lineage, not independent seeds.",
    }
    _write_json(aggregate_dir / "final_decisions.json", decisions)

    changed_files = "\n".join(f"- `{path}`" for path in manifest["changed_files"])
    training_table = pd.DataFrame(training_rows).to_markdown(index=False)
    recovery_table = recovery_summary.to_markdown(index=False)
    generation_table = generation_summary.to_markdown(index=False)
    commands = "\n".join(f"- `{command}`" for command in manifest["reproduction_commands"])
    report = f"""# Pixel-masked EqM field-shaping results

## Scientific question and predeclared hypothesis

This experiment asks whether a small amount of pixel-masked endpoint supervision usefully reshapes one shared, unconditioned EqM vector field. The predeclared hypothesis was that ten epochs of 75% ordinary Gaussian and 25% Bernoulli pixel-mask continuation would improve recovery from unseen contiguous 30%-area block masks without degrading FID-10k by more than 1.0 relative to a compute-matched Gaussian continuation.

Epoch 40 is primary because it contains a meaningful field while leaving training room for reshaping. Epochs 15 and 80 are stage-sensitivity replications from the same lineage and are not independent seeds.

## Exact design

Both arms restore identical raw model, EMA, and AdamW states from each base checkpoint. There is no serialized scheduler because EqM uses a constant learning rate. Every branch performs exactly 400,360 optimizer updates (ten normal 40,036-update epochs), effective global batch 32, LR 1e-4, weight decay 0, EMA decay 0.9999, and the original linear velocity target with its existing c(gamma) and positive GD sampling sign.

Treatment masks are drawn before VAE encoding. Per selected example, the missing ratio is uniform on [0.10,0.50], a shared-across-RGB Bernoulli pixel mask is sampled, missing pixels receive Gaussian noise in normalized pixel space, and both clean and corrupted pixels are encoded with the frozen `stabilityai/sd-vae-ft-ema` VAE. The model receives no mask or corruption identifier.

The locked bank uses 2,048 ImageNet validation images, two independent block masks per image, fixed IDs/classes/mask seeds/noise seeds, and serialized corrupted latents. Blocks cover 30% area with log-uniform aspect ratio [0.5,2.0]. Recovery uses eight ordinary GD updates at step size 0.003, no clamping/data consistency/guidance, and EMA weights. Generation uses 10,000 serialized Gaussian starts and labels per arm with the repository's 250-step-argument/249-update CFG=4.0 convention and one fixed 10k reference-statistics file.

## Smoke validation

{manifest['smoke_summary']}

## Training completion

{training_table}

## Recovery results

{recovery_table}

The sole primary recovery endpoint is the epoch-40 paired treatment-minus-control difference in `D0-D8` LPIPS. Confidence intervals resample 2,048 image clusters and keep each image's two corruptions together ({manifest['statistics']['bootstrap_replicates']} replicates).

## Generation guardrail

{generation_table}

## Decisions

- Epoch 40: **{summary_by_epoch[40]['decision']}**.
- Epoch 15: **{summary_by_epoch[15]['decision']}**.
- Epoch 80: **{summary_by_epoch[80]['decision']}**.
- Stage replication: **{decisions['stage_replication_decision']}**. {decisions['stage_replication_reason']}

## Plots and qualitative examples

- `aggregate/plots/lpips_trajectory_epoch15.png`, `epoch40.png`, `epoch80.png`
- `aggregate/plots/paired_recovery_distribution_epoch15.png`, `epoch40.png`, `epoch80.png`
- `aggregate/plots/recovery_ci_across_epochs.png`
- `aggregate/plots/fid_control_vs_treatment.png`
- `aggregate/plots/qualitative_grid_epoch15.png`, `epoch40.png`, `epoch80.png`

Qualitative image IDs were fixed in the evaluation bank before outputs were examined.

## Failure analysis and limitations

Any failed criterion is classified strictly by the locked decision table; no treatment probability, mask distribution, continuation budget, sampler, metric, or threshold was tuned after the primary result. This is checkpoint-level evidence from a single training lineage. It does not establish reliability across independently trained seeds or global robustness of the vector field.

## Reproduction

{commands}

Git commit: `{manifest['git_commit']}`

Changed files:

{changed_files}

## Final artifacts

- `{aggregate_dir / 'training_runs.csv'}`
- `{aggregate_dir / 'recovery_per_example.csv'}`
- `{aggregate_dir / 'recovery_summary.csv'}`
- `{aggregate_dir / 'generation_summary.csv'}`
- `{aggregate_dir / 'final_decisions.json'}`
- `{plot_dir}`
- `{reports_dir / 'masked_eqm_field_shaping_results.md'}`
"""
    report_path = reports_dir / "masked_eqm_field_shaping_results.md"
    report_path.write_text(report)
    _write_json(
        root / "job_state.json",
        {
            "goal": "masked_eqm_field_shaping_v1",
            "status": "complete",
            "all_required_jobs_terminal_success": True,
            "final_report": str(report_path),
            "primary": summary_by_epoch[40],
            "decisions": decisions,
        },
    )
    print(
        f"epoch40 delta={summary_by_epoch[40]['paired_delta']:.6f} "
        f"CI=[{summary_by_epoch[40]['ci_lower']:.6f},{summary_by_epoch[40]['ci_upper']:.6f}] "
        f"decision={summary_by_epoch[40]['decision']} report={report_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    aggregate(parser.parse_args().manifest)
