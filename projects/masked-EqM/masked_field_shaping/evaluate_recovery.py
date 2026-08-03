"""Locked 8-update held-out contiguous-block recovery evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from masked_field_shaping.evaluation_common import decode_latents, load_ema_model
from masked_field_shaping.train_continuation import freeze_module


RECORDED_STEPS = (0, 1, 2, 4, 8)


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _crop_batch(images, records, margin: int, output_size: int = 128):
    crops = []
    for image, record in zip(images, records):
        top = max(0, int(record["top"]) - margin)
        left = max(0, int(record["left"]) - margin)
        bottom = min(image.shape[-2], int(record["top"]) + int(record["height"]) + margin)
        right = min(image.shape[-1], int(record["left"]) + int(record["width"]) + margin)
        crop = image[:, top:bottom, left:right]
        crops.append(F.interpolate(crop[None], size=(output_size, output_size), mode="bilinear", align_corners=False)[0])
    return torch.stack(crops)


def _pixel_mask(records, height, width, device):
    masks = torch.zeros((len(records), 1, height, width), device=device)
    for index, record in enumerate(records):
        top, left = int(record["top"]), int(record["left"])
        masks[index, :, top : top + int(record["height"]), left : left + int(record["width"])] = 1
    return masks


def _recreate_corrupted(clean, records):
    outputs = []
    for image, record in zip(clean.cpu(), records):
        generator = torch.Generator(device="cpu").manual_seed(int(record["noise_fill_seed"]))
        fill = torch.randn(image.shape, generator=generator)
        mask = torch.zeros((1, image.shape[-2], image.shape[-1]))
        top, left = int(record["top"]), int(record["left"])
        mask[:, top : top + int(record["height"]), left : left + int(record["width"])] = 1
        outputs.append((1 - mask) * image + mask * fill)
    return torch.stack(outputs)


def evaluate(config_path: str) -> None:
    from diffusers.models import AutoencoderKL
    import lpips

    config = json.loads(Path(config_path).read_text())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / "completion.json"
    if completion_path.exists() and json.loads(completion_path.read_text()).get("status") == "completed":
        print(f"recovery already complete: {output}")
        return
    bank = json.loads(Path(config["bank_specification"]).read_text())
    if bank["evaluation_bank_id"] != config["evaluation_bank_id"]:
        raise RuntimeError("evaluation bank ID mismatch")
    if int(config["evaluation_steps"]) != 8:
        raise RuntimeError("recovery evaluation is locked to exactly eight model updates")
    device = torch.device("cuda")
    model, checkpoint = load_ema_model(
        config["checkpoint"],
        model_name=config["model"],
        image_size=int(config["image_size"]),
        num_classes=int(config["num_classes"]),
        device=device,
    )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{config['vae']}").to(device)
    freeze_module(vae)
    perceptual = lpips.LPIPS(net="alex").to(device)
    freeze_module(perceptual)
    shard_output = output / "shards"
    qualitative_dir = output / "qualitative"
    shard_output.mkdir(exist_ok=True)
    qualitative_dir.mkdir(exist_ok=True)
    qualitative_ids = set(bank["qualitative_image_ids"])
    started = time.time()
    total_rows = 0
    all_fields = []

    for shard_index, shard in enumerate(bank["recovery_shards"]):
        result_path = shard_output / f"metrics_{shard_index:04d}.csv"
        diagnostic_path = shard_output / f"diagnostics_{shard_index:04d}.json"
        if result_path.exists() and diagnostic_path.exists():
            with result_path.open() as handle:
                total_rows += sum(1 for _ in csv.DictReader(handle))
            all_fields.append(json.loads(diagnostic_path.read_text()))
            continue
        payload = torch.load(shard["path"], map_location="cpu")
        records = payload["records"]
        clean = torch.stack(
            [payload["clean_uint8"][record["local_clean_index"]] for record in records]
        ).to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)
        labels = torch.tensor([record["class_label"] for record in records], device=device)
        initial = payload["corrupted_latents"].to(device)
        rows = [
            {
                "base_epoch": config["base_epoch"],
                "arm": config["arm"],
                "image_id": record["image_id"],
                "corruption_draw": record["corruption_draw"],
                "mask_seed": record["mask_seed"],
                "block_coordinates": f"{record['top']}:{record['left']}:{record['height']}:{record['width']}",
                "diverged": False,
            }
            for record in records
        ]
        masked_region = _pixel_mask(records, clean.shape[-2], clean.shape[-1], device)
        xt = initial.clone()
        t = torch.ones((xt.shape[0],), device=device, dtype=xt.dtype)
        field_norms = {}
        clipping_fractions = {}
        decoded_states = {}
        nfe = 0

        with torch.no_grad():
            for step in range(0, 9):
                if step in RECORDED_STEPS:
                    decoded = decode_latents(vae, xt)
                    decoded_states[step] = decoded.detach()
                    clipped = decoded.clamp(-1, 1)
                    clipping = ((decoded < -1) | (decoded > 1)).float().flatten(1).mean(1)
                    clipping_fractions[step] = clipping.cpu().tolist()
                    lp = perceptual(clipped, clean).flatten()
                    whole_mse = (clipped - clean).square().flatten(1).mean(1)
                    masked_mse = (
                        (clipped - clean).square() * masked_region
                    ).sum(dim=(1, 2, 3)) / (masked_region.sum(dim=(1, 2, 3)) * clean.shape[1])
                    crop_lp = perceptual(
                        _crop_batch(clipped, records, int(config["crop_context_margin"])),
                        _crop_batch(clean, records, int(config["crop_context_margin"])),
                    ).flatten()
                    for index, row in enumerate(rows):
                        row[f"d{step}_lpips"] = float(lp[index])
                        row[f"whole_image_mse_step{step}"] = float(whole_mse[index])
                        row[f"masked_region_mse_step{step}"] = float(masked_mse[index])
                        row[f"crop_lpips_step{step}"] = float(crop_lp[index])
                        row[f"pixel_clipping_fraction_step{step}"] = float(clipping[index])

                # Field at step 8 is a diagnostic-only ninth evaluation; no
                # update follows it. The sampler still performs exactly 8 updates.
                field = model(xt, t, labels)
                if not torch.is_tensor(field):
                    field = field[0]
                field = field.detach()
                nfe += 1
                norms = field.float().flatten(1).norm(dim=1)
                field_norms[step] = norms.cpu().tolist()
                for index, row in enumerate(rows):
                    row[f"gradient_norm_step{step}"] = float(norms[index])
                    if not torch.isfinite(field[index]).all():
                        row["diverged"] = True
                if step == 8:
                    break
                xt = (xt + field * float(config["sampler_step_size"])).detach()
                t = t + float(config["sampler_step_size"])

        for row in rows:
            row["lpips_recovery"] = row["d0_lpips"] - row["d8_lpips"]
            row["whole_image_mse_step8"] = row["whole_image_mse_step8"]
            row["masked_region_mse_step8"] = row["masked_region_mse_step8"]
            row["gradient_norm_step0"] = row["gradient_norm_step0"]
            row["gradient_norm_step8"] = row["gradient_norm_step8"]

        # The oracle output equals the clean target, so oracle LPIPS is exactly
        # zero and oracle recovery equals D0. This validates metric headroom.
        oracle_mean_recovery = sum(row["d0_lpips"] for row in rows) / len(rows)
        diagnostic = {
            "shard": shard_index,
            "rows": len(rows),
            "updates": 8,
            "model_evaluations": nfe,
            "oracle_mean_recovery": oracle_mean_recovery,
            "diverged": sum(bool(row["diverged"]) for row in rows),
            "mean_clipping_fraction_step8": sum(clipping_fractions[8]) / len(rows),
        }
        temporary = result_path.with_suffix(".csv.tmp")
        fieldnames = list(rows[0].keys())
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, result_path)
        _write_json(diagnostic_path, diagnostic)
        all_fields.append(diagnostic)
        total_rows += len(rows)

        for record_index, record in enumerate(records):
            if record["image_id"] not in qualitative_ids or int(record["corruption_draw"]) != 0:
                continue
            prefix = f"image{record['image_id']:05d}"
            clean_image = clean[record_index].cpu()
            corrupted = _recreate_corrupted(clean[record_index : record_index + 1], [record])[0]
            TF.to_pil_image((clean_image + 1) / 2).save(qualitative_dir / f"{prefix}_clean.png")
            TF.to_pil_image(((corrupted + 1) / 2).clamp(0, 1)).save(
                qualitative_dir / f"{prefix}_corrupted.png"
            )
            TF.to_pil_image(((decoded_states[8][record_index].cpu() + 1) / 2).clamp(0, 1)).save(
                qualitative_dir / f"{prefix}_{config['arm']}_step8.png"
            )
        print(f"recovery shard {shard_index + 1}/{len(bank['recovery_shards'])}", flush=True)

    if total_rows != int(config["recovery_sample_count"]) * int(config["recovery_draws_per_image"]):
        raise RuntimeError(f"recovery row count mismatch: {total_rows}")
    combined_path = output / "recovery_per_example.csv"
    temporary = combined_path.with_suffix(".csv.tmp")
    writer = None
    with temporary.open("w", newline="") as destination:
        for shard_index in range(len(bank["recovery_shards"])):
            with (shard_output / f"metrics_{shard_index:04d}.csv").open() as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)
    os.replace(temporary, combined_path)
    _write_json(
        completion_path,
        {
            "status": "completed",
            "checkpoint": config["checkpoint"],
            "evaluation_bank_id": config["evaluation_bank_id"],
            "rows": total_rows,
            "updates_per_example": 8,
            "model_evaluations_per_example": 9,
            "runtime_seconds": time.time() - started,
            "diverged": sum(item["diverged"] for item in all_fields),
            "output": str(combined_path),
        },
    )
    print(f"completed recovery rows={total_rows} output={combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    evaluate(parser.parse_args().config)
