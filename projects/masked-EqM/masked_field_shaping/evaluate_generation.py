"""Locked paired 10k Gaussian generation and FID guardrail evaluation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image

from masked_field_shaping.evaluation_common import cfg_field, decode_latents, load_ema_model
from masked_field_shaping.train_continuation import freeze_module


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def evaluate(config_path: str) -> None:
    from diffusers.models import AutoencoderKL
    from pytorch_fid.fid_score import calculate_fid_given_paths

    config = json.loads(Path(config_path).read_text())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / "completion.json"
    if completion_path.exists() and json.loads(completion_path.read_text()).get("status") == "completed":
        print(f"generation already complete: {output}")
        return
    bank = json.loads(Path(config["bank_specification"]).read_text())
    if bank["evaluation_bank_id"] != config["evaluation_bank_id"]:
        raise RuntimeError("evaluation bank ID mismatch")
    if sum(int(shard["count"]) for shard in bank["generation_shards"]) != int(config["fid_sample_count"]):
        raise RuntimeError("locked generation input count mismatch")
    device = torch.device("cuda")
    model, _checkpoint = load_ema_model(
        config["checkpoint"],
        model_name=config["model"],
        image_size=int(config["image_size"]),
        num_classes=int(config["num_classes"]),
        device=device,
    )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{config['vae']}").to(device)
    freeze_module(vae)
    generated_dir = output / "generated"
    shard_dir = output / "shards"
    generated_dir.mkdir(exist_ok=True)
    shard_dir.mkdir(exist_ok=True)
    updates = int(config["num_sampling_steps"]) - 1  # exact sample_gd.py convention
    if updates <= 0:
        raise ValueError("num_sampling_steps must be at least 2")
    trajectory_sum = torch.zeros(updates, dtype=torch.float64)
    total_samples = 0
    failed_samples = 0
    started = time.time()

    for shard_index, shard in enumerate(bank["generation_shards"]):
        marker = shard_dir / f"summary_{shard_index:04d}.json"
        if marker.exists():
            existing = json.loads(marker.read_text())
            if existing.get("status") == "completed":
                trajectory_sum += torch.tensor(existing["gradient_norm_sum"], dtype=torch.float64)
                total_samples += int(existing["samples"])
                failed_samples += int(existing["failed_samples"])
                continue
        payload = torch.load(shard["path"], map_location="cpu")
        shard_trajectory = torch.zeros(updates, dtype=torch.float64)
        shard_failures = 0
        batch_size = int(config["batch_size"])
        for start in range(0, int(shard["count"]), batch_size):
            count = min(batch_size, int(shard["count"]) - start)
            xt = payload["noise"][start : start + count].to(device)
            labels = payload["labels"][start : start + count].to(device)
            t = torch.ones((count,), device=device, dtype=xt.dtype)
            with torch.no_grad():
                for update in range(updates):
                    field = cfg_field(
                        model,
                        xt,
                        t,
                        labels,
                        float(config["cfg_scale"]),
                        int(config["num_classes"]),
                    ).detach()
                    finite = torch.isfinite(field).flatten(1).all(1)
                    shard_failures += int((~finite).sum().item())
                    if not finite.all():
                        raise FloatingPointError(
                            f"non-finite generation field in shard {shard_index}, update {update}"
                        )
                    norms = field.float().flatten(1).norm(dim=1)
                    shard_trajectory[update] += norms.double().sum().cpu()
                    xt = (xt + field * float(config["sampler_step_size"])).detach()
                    t = t + float(config["sampler_step_size"])
                decoded = decode_latents(vae, xt)
                decoded = torch.clamp(127.5 * decoded + 128.0, 0, 255)
                decoded = decoded.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            for local_index, image in enumerate(decoded):
                output_index = int(payload["offset"]) + start + local_index
                Image.fromarray(image).save(generated_dir / f"{output_index:06d}.png")
        summary = {
            "status": "completed",
            "samples": int(shard["count"]),
            "failed_samples": shard_failures,
            "gradient_norm_sum": shard_trajectory.tolist(),
            "function_evaluations_per_sample": updates,
        }
        _write_json(marker, summary)
        trajectory_sum += shard_trajectory
        total_samples += int(shard["count"])
        failed_samples += shard_failures
        print(f"generation shard {shard_index + 1}/{len(bank['generation_shards'])}", flush=True)

    present = len(list(generated_dir.glob("*.png")))
    if total_samples != int(config["fid_sample_count"]) or present != int(config["fid_sample_count"]):
        raise RuntimeError(f"generation count mismatch: summaries={total_samples} pngs={present}")
    if failed_samples:
        raise FloatingPointError(f"generation produced {failed_samples} failed sample trajectories")
    fid = calculate_fid_given_paths(
        [bank["generation_reference_stats"], str(generated_dir)],
        batch_size=int(config.get("fid_batch_size", 64)),
        device=device,
        dims=2048,
    )
    runtime = time.time() - started
    summary = {
        "status": "completed",
        "base_epoch": config["base_epoch"],
        "arm": config["arm"],
        "checkpoint": config["checkpoint"],
        "generated_samples": total_samples,
        "fid": float(fid),
        "kid_if_available": None,
        "kid_note": "The repository does not provide KID in its standard generation pipeline.",
        "failed_samples": failed_samples,
        "runtime_seconds": runtime,
        "function_evaluations_per_sample": updates,
        "num_sampling_steps_argument": int(config["num_sampling_steps"]),
        "sampler": "gd",
        "sampler_step_size": float(config["sampler_step_size"]),
        "cfg_scale": float(config["cfg_scale"]),
        "average_gradient_norm_trajectory": (trajectory_sum / total_samples).tolist(),
        "evaluation_bank_id": config["evaluation_bank_id"],
    }
    _write_json(output / "generation_summary.json", summary)
    _write_json(completion_path, summary)
    print(f"FID-10k={fid:.6f} samples={total_samples} output={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    evaluate(parser.parse_args().config)
