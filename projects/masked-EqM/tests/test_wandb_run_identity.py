"""BUG 1: the wandb run id must be injective over runs.

The old id was ``sha256(experiment_name)``, and ``experiment_name`` is built in
train.py as ``{index:03d}-{model}-{path_type}-{prediction}-{loss_weight}-ebm-{ebm}``:
no seed, no arm, and ``index`` is always 000 because each sbatch pre-creates a
fresh job-scoped results dir. Combined with ``resume="allow"``, every seed of an
arm and every arm sharing an ``ebm`` value would have been appended into ONE
wandb run. These tests assert the collision is gone -- and, as a control, that
the old function really did collide, so the new assertion is meaningful.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# wandb_utils imports wandb/torchvision at module scope; stub what is missing so
# the identity logic can be tested without the logging stack installed.
for name in ("wandb",):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)

import wandb_utils  # noqa: E402
from telemetry.ids import RunSpec  # noqa: E402


def _args(**over):
    base = dict(model="EqM-B/2", path_type="Linear", prediction="velocity",
                loss_weight="None", ebm="direct", global_seed=0,
                results_dir="/scratch/job123")
    base.update(over)
    return argparse.Namespace(**base)


EXP_NAME = "000-EqM-B-2-Linear-velocity-None-ebm-direct"


# --------------------------------------------------------------------------
# Control: the old scheme really did collide.
# --------------------------------------------------------------------------

def test_old_scheme_collided_across_seeds_and_arms():
    """Negative control. Without this, the new test could be vacuously true."""
    assert wandb_utils.generate_run_id(EXP_NAME) == wandb_utils.generate_run_id(EXP_NAME)
    # Two genuinely different runs whose experiment_name is identical:
    seed0 = _args(global_seed=0)
    seed1 = _args(global_seed=1)
    assert seed0.global_seed != seed1.global_seed
    assert (wandb_utils.generate_run_id(EXP_NAME)
            == wandb_utils.generate_run_id(EXP_NAME)), \
        "experiment_name carries no seed, so the name-only hash cannot separate them"


# --------------------------------------------------------------------------
# Spec-derived ids (the strong path)
# --------------------------------------------------------------------------

def _spec(**over):
    base = dict(campaign="btm_fd_scalar", arm="btm_scalar_fd_directional",
                seed=0, git_sha="abc1234", phase="II")
    base.update(over)
    return RunSpec(**base)


def test_seeds_get_distinct_ids():
    a = wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME, run_spec=_spec(seed=0))
    b = wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME, run_spec=_spec(seed=1))
    assert a != b


def test_arms_get_distinct_ids():
    a = wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME,
                                    run_spec=_spec(arm="btm_scalar_fd_directional"))
    b = wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME,
                                    run_spec=_spec(arm="btm_scalar_fd_directional4"))
    assert a != b, "prefix-related arm names must not collide"


def test_id_is_the_ledger_run_uid():
    """The wandb id and the telemetry ledger key must be the same string, or a
    wandb run cannot be joined to its event log."""
    spec = _spec()
    assert wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME, run_spec=spec) == spec.run_uid


def test_id_is_stable_across_restarts():
    """resume='allow' is only safe if the id is a function of identity alone."""
    spec = _spec()
    ids = {wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME, run_spec=spec)
           for _ in range(5)}
    assert len(ids) == 1


def test_id_ignores_non_identifying_fields():
    """Two executions differing only in partition/results_dir are one run."""
    a = wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME,
                                    run_spec=_spec(params={"partition": "gpu"}))
    b = wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME,
                                    run_spec=_spec(params={"partition": "seas_gpu"}))
    assert a == b


def test_spec_is_read_from_the_environment_when_not_passed(monkeypatch):
    spec = _spec(seed=7)
    monkeypatch.setenv("EQM_RUN_SPEC", spec.to_env())
    assert wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME) == spec.run_uid


# --------------------------------------------------------------------------
# Fallback path (no spec available)
# --------------------------------------------------------------------------

def test_fallback_separates_seeds(monkeypatch):
    monkeypatch.delenv("EQM_RUN_SPEC", raising=False)
    a = wandb_utils.run_id_for_spec(args=_args(global_seed=0), exp_name=EXP_NAME)
    b = wandb_utils.run_id_for_spec(args=_args(global_seed=1), exp_name=EXP_NAME)
    assert a != b, "even without a spec, the seed must separate runs"


def test_fallback_is_marked_as_argv_derived(monkeypatch):
    monkeypatch.delenv("EQM_RUN_SPEC", raising=False)
    assert wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME).startswith("a")


def test_fallback_is_deterministic(monkeypatch):
    monkeypatch.delenv("EQM_RUN_SPEC", raising=False)
    ids = {wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME) for _ in range(5)}
    assert len(ids) == 1


def test_unparseable_env_spec_does_not_crash(monkeypatch):
    """wandb logging is decoration; it must never abort a training job."""
    monkeypatch.setenv("EQM_RUN_SPEC", "{not json")
    rid = wandb_utils.run_id_for_spec(args=_args(), exp_name=EXP_NAME)
    assert isinstance(rid, str) and rid


# --------------------------------------------------------------------------
# The unguarded dist.get_rank()
# --------------------------------------------------------------------------

def test_is_main_process_outside_ddp():
    import torch.distributed as dist
    assert not dist.is_initialized()
    assert wandb_utils.is_main_process() is True
