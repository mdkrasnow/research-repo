"""Aggregate the six locked runs, bootstrap clusters, plot, and report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from masked_field_shaping.statistics import classify_result, paired_image_cluster_bootstrap


RECOVERY_REQUIRED_COLUMNS = {
    "base_epoch",
    "arm",
    "image_id",
    "corruption_draw",
    "mask_seed",
    "block_coordinates",
    "d0_lpips",
    "d1_lpips",
    "d2_lpips",
    "d4_lpips",
    "d8_lpips",
    "lpips_recovery",
    "whole_image_mse_step8",
    "masked_region_mse_step8",
    "gradient_norm_step0",
    "gradient_norm_step8",
    "diverged",
}


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_finite(values, label: str) -> None:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0 and np.isfinite(array).all(), f"non-finite or empty {label}")


def _validate_recovery_frame(frame: pd.DataFrame, epoch: int, arm: str, manifest: dict) -> None:
    missing = sorted(RECOVERY_REQUIRED_COLUMNS - set(frame.columns))
    _require(not missing, f"epoch {epoch} {arm} recovery missing columns: {missing}")
    expected_images = int(manifest["evaluation_bank"]["recovery_images"])
    expected_draws = int(manifest["evaluation_bank"]["draws_per_image"])
    expected_rows = expected_images * expected_draws
    _require(len(frame) == expected_rows, f"epoch {epoch} {arm} recovery rows {len(frame)} != {expected_rows}")
    _require(frame[["image_id", "corruption_draw"]].duplicated().sum() == 0, f"epoch {epoch} {arm} duplicate recovery keys")
    counts = frame.groupby("image_id", sort=False).size()
    _require(len(counts) == expected_images, f"epoch {epoch} {arm} image clusters {len(counts)} != {expected_images}")
    _require((counts == expected_draws).all(), f"epoch {epoch} {arm} does not have exactly {expected_draws} draws per image")
    _require(set(frame["base_epoch"].astype(int)) == {epoch}, f"epoch {epoch} {arm} recovery base_epoch mismatch")
    _require(set(frame["arm"].astype(str)) == {arm}, f"epoch {epoch} {arm} recovery arm mismatch")
    diverged = frame["diverged"].astype(str).str.lower().isin({"true", "1", "yes"})
    _require(not diverged.any(), f"epoch {epoch} {arm} recovery contains {int(diverged.sum())} diverged examples")
    numeric = [
        "d0_lpips",
        "d1_lpips",
        "d2_lpips",
        "d4_lpips",
        "d8_lpips",
        "lpips_recovery",
        "whole_image_mse_step8",
        "masked_region_mse_step8",
        "gradient_norm_step0",
        "gradient_norm_step8",
    ]
    for column in numeric:
        _require_finite(frame[column], f"epoch {epoch} {arm} recovery {column}")
    residual = np.abs(frame["lpips_recovery"].to_numpy() - (frame["d0_lpips"] - frame["d8_lpips"]).to_numpy())
    _require(float(residual.max()) <= 1e-7, f"epoch {epoch} {arm} LPIPS recovery identity mismatch")
    completion_path = Path(manifest["runs"][f"epoch{epoch}_{arm}"]["recovery_csv"]).parent / "completion.json"
    completion = json.loads(completion_path.read_text())
    _require(completion.get("status") == "completed", f"epoch {epoch} {arm} recovery completion status is not completed")
    _require(int(completion.get("rows", -1)) == expected_rows, f"epoch {epoch} {arm} recovery completion row mismatch")
    _require(int(completion.get("updates_per_example", -1)) == int(manifest["sampler"]["recovery_updates"]), f"epoch {epoch} {arm} recovery update count mismatch")
    _require(completion.get("evaluation_bank_id") == manifest["evaluation_bank"]["id"], f"epoch {epoch} {arm} recovery bank mismatch")


def _validate_generation(generation: dict, epoch: int, arm: str, config: dict, manifest: dict) -> None:
    prefix = f"epoch {epoch} {arm} generation"
    _require(generation.get("status") == "completed", f"{prefix} status is not completed")
    _require(int(generation.get("base_epoch", -1)) == epoch, f"{prefix} base_epoch mismatch")
    _require(generation.get("arm") == arm, f"{prefix} arm mismatch")
    _require(int(generation.get("generated_samples", -1)) == int(manifest["metrics"]["fid_samples"]), f"{prefix} sample count mismatch")
    _require(int(generation.get("failed_samples", -1)) == 0, f"{prefix} contains failed samples")
    _require_finite([generation.get("fid")], f"{prefix} FID")
    _require(generation.get("evaluation_bank_id") == manifest["evaluation_bank"]["id"], f"{prefix} bank mismatch")
    _require(generation.get("sampler") == manifest["sampler"]["type"], f"{prefix} sampler mismatch")
    _require(float(generation.get("sampler_step_size")) == float(manifest["sampler"]["step_size"]), f"{prefix} step size mismatch")
    _require(int(generation.get("num_sampling_steps_argument", -1)) == int(manifest["sampler"]["generation_steps_argument"]), f"{prefix} sampling-step argument mismatch")
    _require(float(generation.get("cfg_scale")) == float(manifest["sampler"]["cfg_scale"]), f"{prefix} CFG mismatch")
    trajectory = generation.get("average_gradient_norm_trajectory", [])
    _require(len(trajectory) == int(generation.get("function_evaluations_per_sample", -1)), f"{prefix} gradient trajectory length mismatch")
    _require_finite(trajectory, f"{prefix} gradient trajectory")
    _require(generation.get("checkpoint") == config["output_dir"] + "/checkpoints/final.pt", f"{prefix} checkpoint mismatch")


def _training_row(epoch, arm, paths, config, manifest):
    completion = json.loads(Path(paths["training_completion"]).read_text())
    initial = json.loads(Path(paths["training_initial_state"]).read_text())
    metrics = [json.loads(line) for line in Path(paths["training_metrics"]).read_text().splitlines() if line]
    expected_updates = int(manifest["continuation_updates"])
    _require(completion.get("status") == "completed", f"epoch {epoch} {arm} training status is not completed")
    _require(int(completion.get("base_epoch", -1)) == epoch, f"epoch {epoch} {arm} training base_epoch mismatch")
    _require(completion.get("arm") == arm, f"epoch {epoch} {arm} training arm mismatch")
    _require(int(completion.get("optimizer_steps", -1)) == expected_updates, f"epoch {epoch} {arm} optimizer-step mismatch")
    _require_finite([completion.get("final_training_loss")], f"epoch {epoch} {arm} final training loss")
    _require(int(config["continuation_updates"]) == expected_updates, f"epoch {epoch} {arm} config continuation budget mismatch")
    _require(int(config["continuation_epochs"]) == int(manifest["continuation_epochs"]), f"epoch {epoch} {arm} continuation-epoch mismatch")
    _require(initial.get("source_checkpoint") == config["base_checkpoint"], f"epoch {epoch} {arm} source checkpoint mismatch")
    _require(int(initial.get("epoch", -1)) == epoch, f"epoch {epoch} {arm} initial epoch mismatch")
    _require(int(initial.get("world_size", -1)) == 4, f"epoch {epoch} {arm} world size mismatch")
    _require(bool(initial.get("restored_optimizer")) == bool(config["preserve_optimizer_state"]), f"epoch {epoch} {arm} optimizer restoration mismatch")
    _require(initial.get("restored_scheduler") is False and initial.get("scheduler_present") is False, f"epoch {epoch} {arm} scheduler-state record mismatch")
    _require(metrics, f"epoch {epoch} {arm} has no training metrics")
    for column in ("loss", "learning_rate", "updates_per_second", "max_gpu_memory_gb"):
        _require_finite([entry[column] for entry in metrics], f"epoch {epoch} {arm} training {column}")
    logged_gradient_norms = [entry["gradient_norm"] for entry in metrics if entry.get("gradient_norm") is not None]
    _require_finite(logged_gradient_norms, f"epoch {epoch} {arm} training gradient_norm")
    last_logged_update = int(metrics[-1]["continuation_updates"])
    _require(
        0 <= expected_updates - last_logged_update < int(config["log_every"]),
        f"epoch {epoch} {arm} final logged metric is not within one log interval of completion",
    )
    _require(int(completion.get("global_step", -1)) == int(initial["step"]) + expected_updates, f"epoch {epoch} {arm} global-step mismatch")
    _require(Path(completion["final_checkpoint"]).is_file(), f"epoch {epoch} {arm} final checkpoint is missing")
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
    expected_mask = float(config["masked_example_probability"])
    _require(abs(weighted_mask - expected_mask) <= 0.02, f"epoch {epoch} {arm} effective mask probability {weighted_mask} is inconsistent with {expected_mask}")
    if expected_mask:
        _require(float(config["min_mask_ratio"]) <= mean_ratio <= float(config["max_mask_ratio"]), f"epoch {epoch} {arm} realized mask ratio is out of range")
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
        "final_training_loss": completion["final_training_loss"],
        "runtime_hours": completion["runtime_hours"],
        "status": completion["status"],
    }, initial


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
        initial_by_arm = {}
        for arm in ("control", "masked"):
            key = f"epoch{epoch}_{arm}"
            paths = manifest["runs"][key]
            config = json.loads(Path(paths["training_config"]).read_text())
            training_row, initial = _training_row(epoch, arm, paths, config, manifest)
            training_rows.append(training_row)
            initial_by_arm[arm] = initial
            frame = pd.read_csv(paths["recovery_csv"])
            _validate_recovery_frame(frame, epoch, arm, manifest)
            recovery_frames.append(frame)
            arm_frames[arm] = frame
            generation = json.loads(Path(paths["generation_summary"]).read_text())
            _validate_generation(generation, epoch, arm, config, manifest)
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

        for fingerprint in ("model_sha256", "ema_sha256", "optimizer_sha256"):
            _require(
                initial_by_arm["control"].get(fingerprint) == initial_by_arm["masked"].get(fingerprint),
                f"epoch {epoch} paired initial {fingerprint} mismatch",
            )
        _require(
            initial_by_arm["control"].get("step") == initial_by_arm["masked"].get("step"),
            f"epoch {epoch} paired initial optimizer-step mismatch",
        )

        keys = ["image_id", "corruption_draw"]
        paired = arm_frames["control"].merge(
            arm_frames["masked"], on=keys, suffixes=("_control", "_treatment"), validate="one_to_one"
        )
        max_d0 = float(np.max(np.abs(paired["d0_lpips_control"] - paired["d0_lpips_treatment"])))
        if max_d0 > float(manifest["metrics"]["d0_tolerance"]):
            raise RuntimeError(f"epoch {epoch} paired D0 mismatch: max abs {max_d0}")
        _require(
            (paired["mask_seed_control"].astype(str) == paired["mask_seed_treatment"].astype(str)).all(),
            f"epoch {epoch} paired mask-seed mismatch",
        )
        _require(
            (paired["block_coordinates_control"].astype(str) == paired["block_coordinates_treatment"].astype(str)).all(),
            f"epoch {epoch} paired block-coordinate mismatch",
        )
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

    try:
        execution_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        execution_changed = subprocess.check_output(
            ["git", "diff", "--name-only", "7aaef1f^", execution_commit], text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        execution_commit = "unavailable"
        execution_changed = []
    changed_file_list = sorted(set(manifest["changed_files"]) | set(execution_changed))
    changed_files = "\n".join(f"- `{path}`" for path in changed_file_list)
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

Frozen scientific-code commit: `{manifest['git_commit']}`

Aggregation execution commit: `{execution_commit}`

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
