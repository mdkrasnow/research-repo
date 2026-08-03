"""Generate deterministic smoke/full arm configurations from the locked protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
CONTROL_ROOT = PROJECT / "experiments" / "masked_eqm_field_shaping"
REMOTE_ROOT = "/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/masked_eqm_field_shaping_v1"
TRAIN = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train"
VAL = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/val"
BASES = {
    15: "/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer15_none_seed0_job35524599/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch15.pt",
    40: "/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer40_none_seed0_ckpt50k_job36359207/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch40.pt",
    80: "/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer80_none_seed0_ckpt50k_job36632776/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch80.pt",
}


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def training_config(base_epoch: int, arm: str, smoke: bool) -> dict:
    masked = arm == "masked"
    stage = "smoke" if smoke else f"epoch{base_epoch}"
    return {
        "experiment_id": "masked_eqm_field_shaping_v1",
        "arm": arm,
        "base_checkpoint": BASES[base_epoch],
        "base_epoch": base_epoch,
        "output_dir": f"{REMOTE_ROOT}/{stage}/{arm}/training",
        "data_path": TRAIN,
        "expected_training_images": 1_281_167,
        "continuation_epochs": 10,
        "continuation_updates": 100 if smoke else 400_360,
        "corruption_mode": "gaussian_or_pixel_mask" if masked else "gaussian",
        "masked_example_probability": 0.25 if masked else 0.0,
        "min_mask_ratio": 0.10,
        "max_mask_ratio": 0.50,
        "mask_type": "bernoulli_pixel",
        "mask_fill": "gaussian_noise",
        "training_seed": 0,
        "mask_seed": 2026080317,
        "preserve_optimizer_state": True,
        "model": "EqM-B/2",
        "image_size": 256,
        "num_classes": 1000,
        "vae": "ema",
        "global_batch_size": 32,
        "per_device_batch_size": 4 if smoke else 8,
        "gradient_accumulation_steps": 8 if smoke else 1,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "ema_decay": 0.9999,
        "path_type": "Linear",
        "prediction": "velocity",
        "loss_weight": None,
        "num_workers": 4,
        "log_every": 10 if smoke else 50,
        "gradient_log_every": 10 if smoke else 200,
        "checkpoint_every": 50 if smoke else 50_000,
        "smoke": smoke,
    }


def bank_config(smoke: bool) -> dict:
    name = "smoke/evaluation_bank" if smoke else "evaluation_bank"
    return {
        "experiment_id": "masked_eqm_field_shaping_v1",
        "evaluation_bank_path": f"{REMOTE_ROOT}/{name}",
        "training_data_path": TRAIN,
        "validation_data_path": VAL,
        "image_size": 256,
        "num_classes": 1000,
        "vae": "ema",
        "evaluation_seed": 2026080301,
        "generation_seed": 2026080302,
        "recovery_sample_count": 16 if smoke else 2048,
        "recovery_draws_per_image": 2,
        "block_area_fraction": 0.30,
        "qualitative_image_count": 4 if smoke else 8,
        "bank_shard_images": 8 if smoke else 32,
        "fid_sample_count": 32 if smoke else 10_000,
        "generation_shard_size": 32 if smoke else 500,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-full", action="store_true")
    args = parser.parse_args()
    configs = CONTROL_ROOT / "configs"
    for arm in ("control", "masked"):
        write(configs / f"smoke_epoch15_{arm}.json", training_config(15, arm, True))
    write(configs / "smoke_evaluation_bank.json", bank_config(True))
    if args.include_full:
        for epoch in (15, 40, 80):
            for arm in ("control", "masked"):
                write(configs / f"epoch{epoch}_{arm}.json", training_config(epoch, arm, False))
        write(configs / "evaluation_bank.json", bank_config(False))


if __name__ == "__main__":
    main()
