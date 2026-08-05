"""Materialize one shared step-0 LPIPS value for every locked recovery draw."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

KEY_FIELDS = ("image_id", "corruption_draw", "mask_seed", "block_coordinates")


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*KEY_FIELDS, "d0_lpips"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def materialize(config_path: str) -> None:
    from diffusers.models import AutoencoderKL
    import lpips
    import torch
    from masked_field_shaping.evaluation_common import decode_latents
    from masked_field_shaping.train_continuation import freeze_module

    config = json.loads(Path(config_path).read_text())
    bank = json.loads(Path(config["bank_specification"]).read_text())
    if bank["evaluation_bank_id"] != config["evaluation_bank_id"]:
        raise RuntimeError("evaluation bank ID mismatch")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / "completion.json"
    expected_rows = int(config["recovery_sample_count"]) * int(config["recovery_draws_per_image"])
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if completion.get("status") == "completed" and completion.get("rows") == expected_rows:
            print(f"shared D0 already complete: {output}")
            return

    device = torch.device("cuda")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{config['vae']}").to(device)
    freeze_module(vae)
    perceptual = lpips.LPIPS(net="alex").to(device)
    freeze_module(perceptual)
    rows: list[dict] = []
    with torch.no_grad():
        for shard_index, shard in enumerate(bank["recovery_shards"]):
            payload = torch.load(shard["path"], map_location="cpu")
            records = payload["records"]
            clean = torch.stack(
                [payload["clean_uint8"][record["local_clean_index"]] for record in records]
            ).to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)
            decoded = decode_latents(vae, payload["corrupted_latents"].to(device))
            values = perceptual((decoded + 1.0) / 2.0, (clean + 1.0) / 2.0).flatten().cpu().tolist()
            for record, value in zip(records, values, strict=True):
                rows.append(
                    {
                        "image_id": int(record["image_id"]),
                        "corruption_draw": int(record["corruption_draw"]),
                        "mask_seed": int(record["mask_seed"]),
                        "block_coordinates": f"{record['top']}:{record['left']}:{record['height']}:{record['width']}",
                        "d0_lpips": float(value),
                    }
                )
            print(f"shared D0 shard {shard_index + 1}/{len(bank['recovery_shards'])}", flush=True)
    if len(rows) != expected_rows:
        raise RuntimeError(f"shared D0 row count {len(rows)} != {expected_rows}")
    keys = {tuple(row[field] for field in KEY_FIELDS) for row in rows}
    if len(keys) != expected_rows:
        raise RuntimeError("shared D0 keys are not unique")
    rows.sort(key=lambda row: (row["image_id"], row["corruption_draw"]))
    csv_path = output / "shared_d0.csv"
    _atomic_csv(csv_path, rows)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    _atomic_json(
        completion_path,
        {
            "status": "completed",
            "rows": expected_rows,
            "evaluation_bank_id": config["evaluation_bank_id"],
            "csv_sha256": digest,
            "metric": "LPIPS-Alex at the common encoded corrupted state before sampler updates",
        },
    )
    print(f"shared D0 completed rows={expected_rows} output={csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    materialize(parser.parse_args().config)
