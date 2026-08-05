import csv
import json

import pytest

from masked_field_shaping.apply_shared_d0 import apply


FIELDS = ["image_id", "corruption_draw", "mask_seed", "block_coordinates", "d0_lpips", "d8_lpips", "lpips_recovery"]


def _write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_apply_shared_d0_replaces_only_common_initial_metric(tmp_path):
    shared = tmp_path / "shared.csv"
    _write(
        shared,
        [
            {"image_id": "1", "corruption_draw": "0", "mask_seed": "11", "block_coordinates": "0:0:2:2", "d0_lpips": "0.4", "d8_lpips": "", "lpips_recovery": ""},
            {"image_id": "1", "corruption_draw": "1", "mask_seed": "12", "block_coordinates": "1:1:2:2", "d0_lpips": "0.5", "d8_lpips": "", "lpips_recovery": ""},
        ],
    )
    directories = {}
    for arm, d8 in (("control", (0.3, 0.35)), ("masked", (0.2, 0.45))):
        directory = tmp_path / arm
        directory.mkdir()
        directories[arm] = str(directory)
        _write(
            directory / "recovery_per_example.csv",
            [
                {"image_id": "1", "corruption_draw": "0", "mask_seed": "11", "block_coordinates": "0:0:2:2", "d0_lpips": "0.1", "d8_lpips": str(d8[0]), "lpips_recovery": "-0.2"},
                {"image_id": "1", "corruption_draw": "1", "mask_seed": "12", "block_coordinates": "1:1:2:2", "d0_lpips": "0.2", "d8_lpips": str(d8[1]), "lpips_recovery": "-0.1"},
            ],
        )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"shared_d0_path": str(shared), "recovery_sample_count": 1, "recovery_draws_per_image": 2, "recovery_output_dirs": directories}))
    apply(str(config))
    for arm, expected in (("control", (0.1, 0.15)), ("masked", (0.2, 0.05))):
        with (tmp_path / arm / "recovery_per_example.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [float(row["d0_lpips"]) for row in rows] == [0.4, 0.5]
        assert [float(row["lpips_recovery"]) for row in rows] == pytest.approx(expected)
        assert (tmp_path / arm / "recovery_per_example_raw_d0.csv").exists()
