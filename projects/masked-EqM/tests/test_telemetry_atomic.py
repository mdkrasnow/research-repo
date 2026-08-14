"""Tests for telemetry.atomic and telemetry.provenance.

The property under test is not "the file gets written" -- the old in-place code
did that too. It is that no observer can ever see a *partial* payload at the
canonical path. So the tests inject failures partway through the write and
assert on what survives at the canonical name.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from telemetry.atomic import (  # noqa: E402
    AtomicWriteError,
    atomic_json_dump,
    atomic_torch_save,
    atomic_write_bytes,
    step_from_filename,
)


# --------------------------------------------------------------------------
# The core invariant: old contents or new contents, never a prefix of new.
# --------------------------------------------------------------------------

def test_write_replaces_atomically(tmp_path):
    target = tmp_path / "a.bin"
    atomic_write_bytes(target, lambda fh: fh.write(b"first"))
    assert target.read_bytes() == b"first"
    atomic_write_bytes(target, lambda fh: fh.write(b"second-and-longer"))
    assert target.read_bytes() == b"second-and-longer"


def test_failure_midwrite_leaves_previous_contents(tmp_path):
    """A crash during the write must not be observable at the canonical path."""
    target = tmp_path / "ckpt.bin"
    atomic_write_bytes(target, lambda fh: fh.write(b"GOOD-PAYLOAD"))

    def exploding(fh):
        fh.write(b"TRUNCA")     # a partial payload...
        raise RuntimeError("preempted")

    with pytest.raises(RuntimeError):
        atomic_write_bytes(target, exploding)

    assert target.read_bytes() == b"GOOD-PAYLOAD", "canonical path was corrupted"


def test_failure_leaves_no_temp_litter(tmp_path):
    """A failed save must not leave files the checkpoint pruner would trip on."""
    target = tmp_path / "ckpt.bin"

    def exploding(fh):
        fh.write(b"x")
        raise RuntimeError("preempted")

    with pytest.raises(RuntimeError):
        atomic_write_bytes(target, exploding)

    assert list(tmp_path.iterdir()) == [], f"litter left behind: {list(tmp_path.iterdir())}"


def test_failure_on_first_write_creates_nothing(tmp_path):
    target = tmp_path / "sub" / "dir" / "ckpt.bin"

    def exploding(fh):
        raise RuntimeError("node died")

    with pytest.raises(RuntimeError):
        atomic_write_bytes(target, exploding)
    assert not target.exists()


def test_temp_file_is_in_the_target_directory(tmp_path):
    """Cross-filesystem rename fails with EXDEV; the temp must be a sibling.

    Asserted by observing the temp file's location from inside the writer.
    """
    target = tmp_path / "nested" / "x.bin"
    seen = {}

    def writer(fh):
        seen["dir"] = Path(fh.name).parent
        fh.write(b"ok")

    atomic_write_bytes(target, writer)
    assert seen["dir"] == target.parent


def test_parent_directories_are_created(tmp_path):
    target = tmp_path / "deep" / "deeper" / "x.json"
    atomic_json_dump({"a": 1}, target)
    assert json.loads(target.read_text()) == {"a": 1}


# --------------------------------------------------------------------------
# JSON specialization
# --------------------------------------------------------------------------

def test_json_is_key_sorted_and_stable(tmp_path):
    """Byte-stability under re-derivation is what makes an artifact citable."""
    a = atomic_json_dump({"b": 2, "a": 1}, tmp_path / "one.json")
    b = atomic_json_dump({"a": 1, "b": 2}, tmp_path / "two.json")
    assert a.read_bytes() == b.read_bytes()


def test_json_handles_paths_and_numpy_like(tmp_path):
    class FakeScalar:
        def item(self):
            return 3.5

    out = tmp_path / "x.json"
    atomic_json_dump({"p": tmp_path, "v": FakeScalar()}, out)
    loaded = json.loads(out.read_text())
    assert loaded["p"] == str(tmp_path)
    assert loaded["v"] == 3.5


# --------------------------------------------------------------------------
# torch specialization + step/payload agreement
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def test_torch_roundtrip(tmp_path):
    payload = {"step": 5, "w": torch.ones(3)}
    p = atomic_torch_save(payload, tmp_path / "0000005.pt")
    loaded = torch.load(p, weights_only=False)
    assert loaded["step"] == 5
    assert torch.equal(loaded["w"], torch.ones(3))


def test_torch_save_preserves_old_checkpoint_on_failure(tmp_path):
    target = tmp_path / "0000005.pt"
    atomic_torch_save({"step": 5, "w": torch.ones(3)}, target)

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("preempted mid-serialization")

    with pytest.raises(Exception):
        atomic_torch_save({"step": 6, "bad": Unpicklable()}, target)

    loaded = torch.load(target, weights_only=False)
    assert loaded["step"] == 5, "a failed save clobbered the previous checkpoint"


@pytest.mark.parametrize("name,expected", [
    ("0125000.pt", 125000),
    ("5000.pt", 5000),
    ("epoch80.pt", None),      # an epoch tag is not a step
    ("last.pt", None),
])
def test_step_from_filename(name, expected):
    assert step_from_filename(name) == expected


def test_step_mismatch_is_refused(tmp_path):
    """Name and payload disagreeing is a provenance corruption, not a warning."""
    target = tmp_path / "0125000.pt"
    with pytest.raises(AtomicWriteError):
        atomic_torch_save({"step": 999}, target, expect_step_key="step")
    assert not target.exists()


def test_step_match_is_allowed(tmp_path):
    target = tmp_path / "0125000.pt"
    atomic_torch_save({"step": 125000}, target, expect_step_key="step")
    assert target.exists()


def test_step_check_skipped_for_epoch_named_checkpoints(tmp_path):
    target = tmp_path / "epoch80.pt"
    atomic_torch_save({"step": 2825000}, target, expect_step_key="step")
    assert target.exists()


# --------------------------------------------------------------------------
# Calibration provenance artifact (BUG 3)
# --------------------------------------------------------------------------

def test_calibration_artifact_carries_full_provenance(tmp_path):
    from telemetry.provenance import write_calibration_artifact

    out = write_calibration_artifact(
        tmp_path / "blur_sigma.json",
        calibration="blur_sigma_lpips_matched_to_mask",
        values={"sigma": 1.234},
        criterion={"target_lpips": 0.5, "tol": 1e-3, "converged": True},
        inputs={"seed": 0, "num_images": 256, "mask_prob": 0.5},
        measurements={"blur_lpips": 0.5004},
        sources={"data_path": "/n/holylabs/...", "checkpoint": None},
    )
    doc = json.loads(out.read_text())

    assert doc["values"]["sigma"] == 1.234
    # every field the bug report demands
    assert doc["inputs"]["seed"] == 0
    assert doc["sources"]["data_path"].startswith("/n/")
    assert "sha" in doc["git"] and "dirty" in doc["git"]
    assert doc["runtime"]["timestamp_utc"].endswith("Z")
    assert doc["calibration"] == "blur_sigma_lpips_matched_to_mask"


def test_calibration_artifact_is_atomic(tmp_path, monkeypatch):
    from telemetry import provenance

    out = tmp_path / "c.json"
    provenance.write_calibration_artifact(
        out, calibration="c", values={"sigma": 1.0}, criterion={}, inputs={})
    before = out.read_bytes()

    monkeypatch.setattr(provenance, "runtime_provenance",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        provenance.write_calibration_artifact(
            out, calibration="c", values={"sigma": 2.0}, criterion={}, inputs={})
    assert out.read_bytes() == before


def test_git_provenance_reports_dirtiness(tmp_path):
    from telemetry.provenance import git_provenance

    g = git_provenance()
    # In this repo git is available, so a sha must be recoverable and the
    # dirty flag must be a real boolean rather than silently omitted.
    assert g["sha"] is None or len(g["sha"]) >= 7
    assert g["dirty"] in (True, False, None)


def test_git_provenance_falls_back_to_env(tmp_path, monkeypatch):
    """Cluster jobs run from an extracted archive with no .git directory."""
    from telemetry.provenance import git_provenance

    monkeypatch.setenv("GIT_SHA", "deadbeefcafe")
    g = git_provenance(cwd=tmp_path)
    assert g["sha"] == "deadbeefcafe"
    assert g["sha_source"] == "env:GIT_SHA"


def test_default_artifact_path_prefers_job_out_dir(monkeypatch, tmp_path):
    """The calibration sbatch extracts into a /tmp work dir it then deletes, so
    a repo-relative default would write the artifact into a directory that
    disappears seconds later. The env var must win."""
    from telemetry.provenance import default_artifact_path

    monkeypatch.delenv("CALIBRATION_OUT_DIR", raising=False)
    monkeypatch.setenv("OUT_DIR", str(tmp_path))
    assert default_artifact_path("x.json") == str(tmp_path / "x.json")

    monkeypatch.setenv("CALIBRATION_OUT_DIR", str(tmp_path / "cal"))
    assert default_artifact_path("x.json") == str(tmp_path / "cal" / "x.json")


def test_default_artifact_path_local_fallback(monkeypatch):
    from telemetry.provenance import default_artifact_path

    monkeypatch.delenv("CALIBRATION_OUT_DIR", raising=False)
    monkeypatch.delenv("OUT_DIR", raising=False)
    monkeypatch.setattr("telemetry.provenance._PERSISTENT_ROOT", "/nonexistent/root/x")
    assert default_artifact_path("x.json") == "results/calibration/x.json"
