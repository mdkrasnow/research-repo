import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.direct_energy.analyze_training_regression import parse_log


def test_parse_training_record(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text("[\x1b[34m2026-07-31 12:00:00\x1b[0m] (step=1517550) Train Loss: 12.1100, Train Steps/Sec: 8.40\n")
    records = parse_log(log)
    assert len(records) == 1
    assert records[0].step == 1_517_550
    assert records[0].loss == 12.11
    assert records[0].steps_per_sec == 8.4
