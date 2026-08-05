"""Apply a common D0 bank to completed per-arm recovery rows without rerunning sampling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from masked_field_shaping.shared_d0 import KEY_FIELDS


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _key(row: dict) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def apply(config_path: str) -> None:
    config = json.loads(Path(config_path).read_text())
    shared_path = Path(config["shared_d0_path"])
    with shared_path.open(newline="") as handle:
        shared_rows = list(csv.DictReader(handle))
    shared = {_key(row): float(row["d0_lpips"]) for row in shared_rows}
    expected_rows = int(config["recovery_sample_count"]) * int(config["recovery_draws_per_image"])
    if len(shared) != expected_rows:
        raise RuntimeError(f"shared D0 keys {len(shared)} != {expected_rows}")
    shared_sha256 = hashlib.sha256(shared_path.read_bytes()).hexdigest()

    for arm, directory in config["recovery_output_dirs"].items():
        output = Path(directory)
        path = output / "recovery_per_example.csv"
        raw_path = output / "recovery_per_example_raw_d0.csv"
        metadata_path = output / "shared_d0_applied.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("shared_d0_sha256") == shared_sha256:
                continue
            raise RuntimeError(f"{arm} already has a different shared D0 application")
        rows = _read_rows(path)
        if len(rows) != expected_rows or {_key(row) for row in rows} != set(shared):
            raise RuntimeError(f"{arm} recovery rows do not exactly match shared D0 bank")
        if not raw_path.exists():
            os.replace(path, raw_path)
            rows = _read_rows(raw_path)
        for row in rows:
            d0 = shared[_key(row)]
            row["d0_lpips"] = repr(d0)
            row["lpips_recovery"] = repr(d0 - float(row["d8_lpips"]))
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
        _atomic_json(
            metadata_path,
            {
                "status": "completed",
                "arm": arm,
                "rows": expected_rows,
                "shared_d0_path": str(shared_path),
                "shared_d0_sha256": shared_sha256,
                "raw_backup": str(raw_path),
                "operation": "replace per-arm separately decoded D0 with one common locked-bank D0; retain sampled D8",
            },
        )
        print(f"applied shared D0 arm={arm} rows={expected_rows}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    apply(parser.parse_args().config)
