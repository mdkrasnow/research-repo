"""Build the immutable validation-only recovery bank and generation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF

from masked_field_shaping.corruption import rectangular_block_mask, sample_block_parameters
from masked_field_shaping.train_continuation import center_crop_arr, freeze_module


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(namespace: str, image_id: int, draw: int) -> int:
    payload = f"masked-eqm-field-shaping-v1:{namespace}:{image_id}:{draw}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _normalized_image(dataset: ImageFolder, index: int, image_size: int):
    path, label = dataset.samples[index]
    image = center_crop_arr(dataset.loader(path), image_size)
    uint8 = torch.from_numpy(__import__("numpy").array(image, copy=True)).permute(2, 0, 1)
    normalized = uint8.float().div(127.5).sub(1.0)
    return uint8, normalized, label, path


def build(config_path: str) -> None:
    from diffusers.models import AutoencoderKL
    from pytorch_fid.fid_score import save_fid_stats

    config = json.loads(Path(config_path).read_text())
    output = Path(config["evaluation_bank_path"])
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if completion.get("status") == "completed" and completion.get("config_sha256") == hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest():
            print(f"evaluation bank already complete: {output}")
            return

    if Path(config["validation_data_path"]).resolve() == Path(config["training_data_path"]).resolve():
        raise RuntimeError("evaluation bank must not use the training directory")
    if "val" not in Path(config["validation_data_path"]).name.lower():
        raise RuntimeError("validation_data_path does not look like the ImageNet validation split")
    dataset = ImageFolder(config["validation_data_path"])
    if len(dataset.classes) != 1000 or len(dataset) != 50_000:
        raise RuntimeError(f"unexpected validation split: classes={len(dataset.classes)} images={len(dataset)}")

    generator = torch.Generator(device="cpu").manual_seed(int(config["evaluation_seed"]))
    permutation = torch.randperm(len(dataset), generator=generator)
    image_ids = permutation[: int(config["recovery_sample_count"])].tolist()
    qualitative_ids = sorted(image_ids[: int(config["qualitative_image_count"])])
    device = torch.device("cuda")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{config['vae']}").to(device)
    freeze_module(vae)

    shard_size = int(config.get("bank_shard_images", 64))
    shard_records = []
    metadata_all = []
    for shard_start in range(0, len(image_ids), shard_size):
        shard_number = shard_start // shard_size
        shard_path = output / f"recovery_shard_{shard_number:04d}.pt"
        shard_meta_path = output / f"recovery_shard_{shard_number:04d}.json"
        if shard_path.exists() and shard_meta_path.exists():
            shard_records.append(
                {"path": str(shard_path), "sha256": _file_sha256(shard_path), "metadata": str(shard_meta_path)}
            )
            metadata_all.extend(json.loads(shard_meta_path.read_text())["records"])
            continue
        batch_ids = image_ids[shard_start : shard_start + shard_size]
        clean_uint8 = []
        labels = []
        relative_paths = []
        corrupted_pixels = []
        records = []
        for local_index, image_id in enumerate(batch_ids):
            clean_u8, clean, label, source_path = _normalized_image(
                dataset, image_id, int(config["image_size"])
            )
            clean_uint8.append(clean_u8)
            labels.append(label)
            relative_paths.append(os.path.relpath(source_path, config["validation_data_path"]))
            for draw in range(int(config["recovery_draws_per_image"])):
                mask_seed = _seed("block-mask", image_id, draw)
                noise_seed = _seed("noise-fill", image_id, draw)
                encode_seed = _seed("vae-encode", image_id, draw)
                aspect, top_uniform, left_uniform = sample_block_parameters(mask_seed)
                block = rectangular_block_mask(
                    int(config["image_size"]),
                    int(config["image_size"]),
                    float(config["block_area_fraction"]),
                    aspect,
                    top_uniform=top_uniform,
                    left_uniform=left_uniform,
                )
                noise_generator = torch.Generator(device="cpu").manual_seed(noise_seed)
                fill = torch.randn(clean.shape, generator=noise_generator)
                corrupted = block.keep_mask * clean + (1.0 - block.keep_mask) * fill
                corrupted_pixels.append(corrupted)
                records.append(
                    {
                        "image_id": image_id,
                        "local_clean_index": local_index,
                        "corruption_draw": draw,
                        "class_label": label,
                        "mask_seed": mask_seed,
                        "noise_fill_seed": noise_seed,
                        "vae_encode_seed": encode_seed,
                        "aspect_ratio": aspect,
                        "top": block.top,
                        "left": block.left,
                        "height": block.height,
                        "width": block.width,
                        "requested_area_fraction": block.requested_area_fraction,
                        "realized_area_fraction": block.realized_area_fraction,
                    }
                )
        corrupted_batch = torch.stack(corrupted_pixels).to(device)
        with torch.no_grad():
            distribution = vae.encode(corrupted_batch).latent_dist
            latents = []
            for row, record in enumerate(records):
                encode_generator = torch.Generator(device=device).manual_seed(record["vae_encode_seed"])
                noise = torch.randn(
                    distribution.mean[row].shape,
                    device=device,
                    dtype=distribution.mean.dtype,
                    generator=encode_generator,
                )
                latents.append(distribution.mean[row] + distribution.std[row] * noise)
            encoded = torch.stack(latents).mul_(0.18215).cpu()
        payload = {
            "clean_uint8": torch.stack(clean_uint8),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_ids": torch.tensor(batch_ids, dtype=torch.long),
            "relative_paths": relative_paths,
            "corrupted_latents": encoded,
            "records": records,
        }
        temporary = shard_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, shard_path)
        _write_json(shard_meta_path, {"records": records})
        shard_records.append(
            {"path": str(shard_path), "sha256": _file_sha256(shard_path), "metadata": str(shard_meta_path)}
        )
        metadata_all.extend(records)
        print(f"built {shard_path} ({len(records)} corruptions)", flush=True)

    if len(metadata_all) != int(config["recovery_sample_count"]) * int(config["recovery_draws_per_image"]):
        raise RuntimeError("evaluation bank corruption count mismatch")
    counts = {}
    for record in metadata_all:
        counts[record["image_id"]] = counts.get(record["image_id"], 0) + 1
    if set(counts.values()) != {2}:
        raise RuntimeError("each evaluation image must have exactly two draws")

    # Locked 10k generation schedule: exact latent noise and class labels are
    # serialized once, so every arm consumes byte-identical inputs.
    generation_dir = output / "generation_inputs"
    generation_dir.mkdir(exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(int(config["generation_seed"]))
    remaining = int(config["fid_sample_count"])
    generation_shards = []
    offset = 0
    while remaining:
        count = min(int(config.get("generation_shard_size", 1000)), remaining)
        shard_path = generation_dir / f"inputs_{offset:05d}_{offset + count:05d}.pt"
        if not shard_path.exists():
            payload = {
                "offset": offset,
                "noise": torch.randn(count, 4, int(config["image_size"]) // 8, int(config["image_size"]) // 8, generator=gen),
                "labels": torch.randint(0, int(config["num_classes"]), (count,), generator=gen),
            }
            temporary = shard_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, shard_path)
        else:
            # Advance the canonical stream even when resuming after a partial build.
            torch.randn(count, 4, int(config["image_size"]) // 8, int(config["image_size"]) // 8, generator=gen)
            torch.randint(0, int(config["num_classes"]), (count,), generator=gen)
        generation_shards.append({"path": str(shard_path), "sha256": _file_sha256(shard_path), "count": count})
        offset += count
        remaining -= count

    # A single locked 10k reference sample and its Inception statistics.
    reference_dir = output / "fid_reference_10k"
    reference_dir.mkdir(exist_ok=True)
    reference_ids = permutation[
        int(config["recovery_sample_count"]) : int(config["recovery_sample_count"]) + int(config["fid_sample_count"])
    ].tolist()
    for out_index, image_id in enumerate(reference_ids):
        path = reference_dir / f"{out_index:06d}.png"
        if path.exists():
            continue
        clean_u8, _clean, _label, _source = _normalized_image(dataset, image_id, int(config["image_size"]))
        TF.to_pil_image(clean_u8).save(path)
    reference_stats = output / "fid_reference_10k_stats.npz"
    if not reference_stats.exists():
        save_fid_stats([str(reference_dir), str(reference_stats)], batch_size=64, device=device, dims=2048)

    spec = {
        "experiment_id": "masked_eqm_field_shaping_v1",
        "validation_data_path": config["validation_data_path"],
        "training_data_path": config["training_data_path"],
        "image_ids": image_ids,
        "qualitative_image_ids": qualitative_ids,
        "recovery_records": len(metadata_all),
        "recovery_shards": shard_records,
        "generation_shards": generation_shards,
        "generation_reference_image_ids": reference_ids,
        "generation_reference_stats": str(reference_stats),
        "vae": config["vae"],
        "latent_scale": 0.18215,
        "block_area_fraction": config["block_area_fraction"],
        "block_aspect_ratio": [0.5, 2.0],
        "config": config,
    }
    bank_id = canonical = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    spec["evaluation_bank_id"] = bank_id
    _write_json(output / "bank_specification.json", spec)
    _write_json(
        completion_path,
        {
            "status": "completed",
            "evaluation_bank_id": bank_id,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "recovery_records": len(metadata_all),
            "generation_inputs": int(config["fid_sample_count"]),
            "reference_stats": str(reference_stats),
        },
    )
    print(f"evaluation_bank_id={bank_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    build(parser.parse_args().config)
