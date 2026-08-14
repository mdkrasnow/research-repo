"""Turn a computed constant into a citable artifact.

The defect this replaces
------------------------
``calibrate_blur_sigma.py`` and ``calibrate_downsample_fourier.py`` each derive a
constant (``sigma``, ``downsample_factor``, ``fourier_cutoff``) that was then
hardcoded into training configs -- and each wrote *no file at all*.  The only
record of where the number came from was a ``RESULT ...`` line in a SLURM stdout
log, i.e. in a rotating, unindexed, unparsed side channel.  A constant that
appears in a training config with no machine-readable derivation is
indistinguishable from a constant somebody guessed, which means every downstream
claim that depends on it is unverifiable in principle.

What an artifact has to carry to close that gap
-----------------------------------------------
The value alone is not provenance.  Provenance is enough information to *re-run
the derivation and check you get the same number*:

* the derived value(s), plus the target/criterion they were matched against;
* every input that changes the result (the full parsed argument namespace,
  including the RNG seed -- these calibrations draw a random subset of the
  validation set, so an unrecorded seed makes the number irreproducible even
  with identical code);
* the artifacts consumed (data path, VAE weights, checkpoint if any);
* the code version (git sha, and whether the tree was dirty -- a clean sha over
  a dirty tree is a *false* provenance claim, worse than none);
* when it was produced, and on what host/job.

The artifact is written with :func:`telemetry.atomic.atomic_json_dump`, so a
preemption during the write cannot leave a half-JSON file that a config
generator would then fail to parse -- or, worse, that a lenient parser would
read as a truncated-but-plausible value.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .atomic import atomic_json_dump

__all__ = ["git_provenance", "runtime_provenance", "write_calibration_artifact",
           "default_artifact_path"]

#: Where cluster jobs persist results. The calibration sbatch scripts extract the
#: source into a /tmp work dir and ``rm -rf`` it on exit, so a repo-relative
#: default would write the artifact into a directory that is deleted seconds
#: later -- the artifact would exist and still be unrecoverable.
_PERSISTENT_ROOT = "/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results"


def default_artifact_path(filename: str) -> str:
    """Best persistent location for ``filename``, resolved at call time.

    Precedence: ``CALIBRATION_OUT_DIR`` -> ``OUT_DIR`` (what the sbatch scripts
    already name) -> the cluster persistent root if it is mounted -> a
    repo-relative path for local runs.  Resolved at runtime rather than baked in
    as an argparse default string so the same script is correct on the cluster
    and on a laptop without a flag.
    """
    for var in ("CALIBRATION_OUT_DIR", "OUT_DIR"):
        value = os.environ.get(var)
        if value:
            return str(Path(value) / filename)
    if Path(_PERSISTENT_ROOT).parent.is_dir():
        return str(Path(_PERSISTENT_ROOT) / filename)
    return str(Path("results") / "calibration" / filename)


def _run_git(args, cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_provenance(cwd: Optional[Path] = None) -> Dict[str, Any]:
    """``{sha, dirty, branch}`` for the tree the code was run from.

    ``dirty`` is recorded rather than suppressed: a sha identifies committed
    code, and reporting a sha while the working tree had uncommitted edits is an
    unsound provenance claim.  A consumer that needs an exactly-reproducible
    derivation should reject artifacts with ``dirty=True``.  Environment
    override ``GIT_SHA`` wins, because cluster jobs run from an extracted source
    archive with no ``.git`` directory but do have the sha exported.
    """
    cwd = Path(cwd or Path(__file__).resolve().parent)
    env_sha = os.environ.get("GIT_SHA")
    sha = _run_git(["rev-parse", "HEAD"], cwd) or env_sha
    status = _run_git(["status", "--porcelain"], cwd)
    return {
        "sha": sha,
        "sha_source": "git" if _run_git(["rev-parse", "HEAD"], cwd) else
                      ("env:GIT_SHA" if env_sha else "unavailable"),
        "dirty": None if status is None else bool(status.strip()),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
    }


def runtime_provenance() -> Dict[str, Any]:
    """When/where the derivation ran, so it can be traced back to a SLURM log."""
    info: Dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "argv": list(sys.argv),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_device"] = (torch.cuda.get_device_name(0)
                               if torch.cuda.is_available() else None)
    except Exception:
        pass
    return info


def write_calibration_artifact(
    path,
    *,
    calibration: str,
    values: Mapping[str, Any],
    criterion: Mapping[str, Any],
    inputs: Mapping[str, Any],
    measurements: Optional[Mapping[str, Any]] = None,
    sources: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write one calibration result as an atomic, self-describing JSON artifact.

    ``values``       the constants that get hardcoded downstream.
    ``criterion``    what they were matched to, and the tolerance (so a reader
                     can tell "converged to 1e-3" from "hit the iteration cap").
    ``inputs``       the full argument namespace, seed included.
    ``measurements`` every measured quantity reported alongside the value.
    ``sources``      data paths / VAE / checkpoints the numbers were derived from.
    """
    document = {
        "artifact": "calibration",
        "artifact_version": 1,
        "calibration": calibration,
        "values": dict(values),
        "criterion": dict(criterion),
        "inputs": dict(inputs),
        "measurements": dict(measurements or {}),
        "sources": dict(sources or {}),
        "git": git_provenance(),
        "runtime": runtime_provenance(),
    }
    return atomic_json_dump(document, path)
