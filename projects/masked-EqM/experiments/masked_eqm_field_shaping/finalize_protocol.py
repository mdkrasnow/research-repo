"""Freeze evaluation configs and the immutable full-run manifest after smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from prepare_configs import BASES, CONTROL_ROOT, REMOTE_ROOT


PROJECT_REL = "projects/masked-EqM"


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def recovery_config(epoch, arm, checkpoint, bank, smoke=False):
    stage = "smoke" if smoke else f"epoch{epoch}"
    return {
        "experiment_id": "masked_eqm_field_shaping_v1",
        "base_epoch": epoch,
        "arm": arm,
        "checkpoint": checkpoint,
        "output_dir": f"{REMOTE_ROOT}/{stage}/{arm}/recovery",
        "bank_specification": bank["spec_path"],
        "evaluation_bank_id": bank["evaluation_bank_id"],
        "evaluation_steps": 8,
        "recovery_sample_count": bank["config"]["recovery_sample_count"],
        "recovery_draws_per_image": 2,
        "model": "EqM-B/2",
        "image_size": 256,
        "num_classes": 1000,
        "vae": "ema",
        "sampler": "gd",
        "sampler_step_size": 0.003,
        "crop_context_margin": 16,
    }


def generation_config(epoch, arm, checkpoint, bank, smoke=False):
    stage = "smoke" if smoke else f"epoch{epoch}"
    return {
        "experiment_id": "masked_eqm_field_shaping_v1",
        "base_epoch": epoch,
        "arm": arm,
        "checkpoint": checkpoint,
        "output_dir": f"{REMOTE_ROOT}/{stage}/{arm}/generation",
        "bank_specification": bank["spec_path"],
        "evaluation_bank_id": bank["evaluation_bank_id"],
        "fid_sample_count": bank["config"]["fid_sample_count"],
        "model": "EqM-B/2",
        "image_size": 256,
        "num_classes": 1000,
        "vae": "ema",
        "sampler": "gd",
        "sampler_step_size": 0.003,
        "num_sampling_steps": 2 if smoke else 250,
        "cfg_scale": 4.0,
        "batch_size": 4 if smoke else 16,
        "fid_batch_size": 32 if smoke else 64,
    }


def task(task_id, kind, config_name, output_dir, completion, depends_on, *, task_name=None):
    training = kind == "training"
    return {
        "id": task_id,
        "kind": kind,
        "task": task_name,
        "config_rel": f"{PROJECT_REL}/experiments/masked_eqm_field_shaping/configs/{config_name}",
        "sbatch": f"{PROJECT_REL}/slurm/jobs/" + (
            "masked_field_shaping_full_train.sbatch" if training else "masked_field_shaping_eval.sbatch"
        ),
        "output_dir": output_dir,
        "completion_marker": completion,
        "depends_on": depends_on,
        "log_pattern": "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/"
        + ("masked-field-full-{job_id}.out" if training else "masked-field-eval-{job_id}.out"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-bank-spec", required=True)
    parser.add_argument("--full-bank-spec", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--smoke-summary", required=True)
    args = parser.parse_args()
    configs = CONTROL_ROOT / "configs"
    smoke_bank = json.loads(Path(args.smoke_bank_spec).read_text())
    full_bank = json.loads(Path(args.full_bank_spec).read_text())
    smoke_bank["spec_path"] = f"{REMOTE_ROOT}/smoke/evaluation_bank/bank_specification.json"
    full_bank["spec_path"] = f"{REMOTE_ROOT}/evaluation_bank/bank_specification.json"

    for arm in ("control", "masked"):
        checkpoint = f"{REMOTE_ROOT}/smoke/{arm}/training/checkpoints/final.pt"
        write(configs / f"smoke_recovery_{arm}.json", recovery_config(15, arm, checkpoint, smoke_bank, True))
        write(configs / f"smoke_generation_{arm}.json", generation_config(15, arm, checkpoint, smoke_bank, True))
    for epoch in (15, 40, 80):
        for arm in ("control", "masked"):
            checkpoint = f"{REMOTE_ROOT}/epoch{epoch}/{arm}/training/checkpoints/final.pt"
            write(configs / f"epoch{epoch}_{arm}_recovery.json", recovery_config(epoch, arm, checkpoint, full_bank))
            write(configs / f"epoch{epoch}_{arm}_generation.json", generation_config(epoch, arm, checkpoint, full_bank))

    workflow = []
    epoch40_eval_ids = []
    for epoch in (40, 15, 80):
        stage_dependencies = epoch40_eval_ids if epoch != 40 else []
        for arm in ("control", "masked"):
            prefix = f"epoch{epoch}_{arm}"
            training_dir = f"{REMOTE_ROOT}/epoch{epoch}/{arm}/training"
            workflow.append(
                task(prefix + "_training", "training", prefix + ".json", training_dir, training_dir + "/completion.json", list(stage_dependencies))
            )
            for evaluation in ("recovery", "generation"):
                evaluation_dir = f"{REMOTE_ROOT}/epoch{epoch}/{arm}/{evaluation}"
                task_id = prefix + "_" + evaluation
                workflow.append(
                    task(
                        task_id,
                        evaluation,
                        prefix + f"_{evaluation}.json",
                        evaluation_dir,
                        evaluation_dir + "/completion.json",
                        [prefix + "_training"],
                        task_name=evaluation,
                    )
                )
                if epoch == 40:
                    epoch40_eval_ids.append(task_id)
    all_eval_ids = [entry["id"] for entry in workflow if entry["kind"] in {"recovery", "generation"}]
    workflow.append(
        task(
            "aggregate",
            "aggregate",
            "../manifest.yaml",
            f"{REMOTE_ROOT}/aggregate",
            f"{REMOTE_ROOT}/reports/masked_eqm_field_shaping_results.md",
            all_eval_ids,
            task_name="aggregate",
        )
    )

    runs = {}
    for epoch in (15, 40, 80):
        for arm in ("control", "masked"):
            root = f"{REMOTE_ROOT}/epoch{epoch}/{arm}"
            runs[f"epoch{epoch}_{arm}"] = {
                "training_config": f"{root}/training/resolved_configuration.json",
                "training_completion": f"{root}/training/completion.json",
                "training_initial_state": f"{root}/training/initial_state.json",
                "training_metrics": f"{root}/training/training_metrics.jsonl",
                "recovery_csv": f"{root}/recovery/recovery_per_example.csv",
                "recovery_qualitative": f"{root}/recovery/qualitative",
                "generation_summary": f"{root}/generation/generation_summary.json",
            }
    changed_files = subprocess.check_output(
        ["git", "diff", "--name-only", "7aaef1f^", args.git_commit], text=True
    ).splitlines()
    manifest = {
        "experiment_id": "masked_eqm_field_shaping_v1",
        "immutable_after": "first full training job submission",
        "git_commit": args.git_commit,
        "artifact_root": REMOTE_ROOT,
        "base_checkpoints": {str(key): value for key, value in BASES.items()},
        "continuation_epochs": 10,
        "continuation_updates": 400_360,
        "training_seed": 0,
        "mask_seed": 2026080317,
        "training_data_path": full_bank["config"]["training_data_path"],
        "validation_data_path": full_bank["config"]["validation_data_path"],
        "scheduler_resources": {
            "training": "seas_gpu, 4xH200, exclusive, 48h, effective global batch 32",
            "evaluation": "gpu_test, one A100 MIG slice, 12h",
        },
        "evaluation_bank": {
            "id": full_bank["evaluation_bank_id"],
            "path": full_bank["spec_path"],
            "qualitative_image_ids": full_bank["qualitative_image_ids"],
            "recovery_images": 2048,
            "draws_per_image": 2,
        },
        "sampler": {"type": "gd", "step_size": 0.003, "recovery_updates": 8, "generation_steps_argument": 250, "cfg_scale": 4.0},
        "metrics": {"primary": "paired LPIPS recovery D0-D8", "d0_tolerance": 1e-6, "fid_samples": 10_000},
        "statistics": {"bootstrap": "paired image-cluster", "bootstrap_replicates": 10_000, "bootstrap_seed": 2026080303},
        "success_criteria": {"recovery_ci_lower_gt": 0.0, "fid_treatment_minus_control_lte": 1.0},
        "smoke_summary": args.smoke_summary,
        "runs": runs,
        "workflow": workflow,
        "changed_files": changed_files,
        "reproduction_commands": [
            "python -m pytest -q tests/test_masked_field_shaping.py",
            "python experiments/masked_eqm_field_shaping/goal_runner.py --manifest experiments/masked_eqm_field_shaping/manifest.yaml",
            "python -m masked_field_shaping.aggregate_results --manifest experiments/masked_eqm_field_shaping/manifest.yaml",
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write(CONTROL_ROOT / "manifest.yaml", manifest)


if __name__ == "__main__":
    main()
