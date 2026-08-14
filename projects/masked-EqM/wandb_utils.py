import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import json
import math


def is_main_process():
    """True on rank 0, and also when there is no process group at all.

    ``dist.get_rank()`` raises outside an initialized process group, so the
    unguarded form made this module unusable from any single-process context
    (``sample_gd.py``, notebooks, CPU smoke tests) -- a logging helper must
    degrade to "I am the only process" rather than crash the caller.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    """DEPRECATED -- name-only hash, retained solely to resolve pre-existing runs.

    This function is not injective over *runs*: ``exp_name`` is built in
    ``train.py`` from ``{experiment_index}-{model}-{path_type}-{prediction}-
    {loss_weight}-ebm-{ebm}``, which contains neither the seed nor the arm, and
    ``experiment_index = len(glob(results_dir/*))`` is always ``000`` because
    every sbatch pre-creates a fresh job-scoped results dir.  So all seeds of an
    arm, and every arm sharing an ``ebm`` value, mapped to ONE id -- and with
    ``resume="allow"`` wandb would have silently appended all of them into a
    single run, producing a step series that interleaves independent trainings.

    Use :func:`run_id_for_spec` instead.  This is kept only so that an operator
    can still compute the id of a historical run when reconciling old wandb data.
    """
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def _spec_from_env():
    """The launcher-stated run identity, or ``None`` if this run has none.

    Never raises: wandb logging is decoration, and a decoration layer must not
    be able to abort a training job.  A missing spec is a real condition (local
    runs, legacy launchers) that the fallback below handles.
    """
    try:
        from telemetry import RunSpec
        return RunSpec.from_env()
    except Exception:
        return None


def run_id_for_spec(args=None, exp_name="", run_spec=None):
    """The wandb run id for this run, derived from its *identity*, not its name.

    Precedence, strongest evidence first:

    1. ``run_spec.run_uid`` -- the content hash of the launcher-stated spec
       (campaign, arm, seed, git sha, params).  This is the same key the
       telemetry ledger joins on, so a wandb run and its event log are trivially
       cross-referenced, and two runs collide in wandb iff they are genuinely
       the same experiment (which is exactly when ``resume="allow"`` is correct).
    2. ``EQM_RUN_SPEC`` from the environment, when the caller did not pass one.
    3. Fallback: a content hash over the *full* argparse namespace plus
       ``exp_name``.  Weaker than (1) -- it has no campaign/arm concept and it
       includes non-identifying fields -- but it does contain ``global_seed`` and
       every arm-selecting flag, so it separates the runs that the old name-only
       hash merged.  Prefixed ``a`` (for "argv-derived") so an id's provenance is
       readable off the id itself.

    The result is stable across restarts of the same run, which is what makes
    ``resume="allow"`` meaningful instead of dangerous.
    """
    spec = run_spec if run_spec is not None else _spec_from_env()
    if spec is not None:
        try:
            return spec.run_uid
        except Exception:
            pass

    payload = {"exp_name": str(exp_name)}
    if args is not None:
        try:
            payload["args"] = namespace_to_dict(args)
        except Exception:
            payload["args"] = repr(args)
    try:
        from telemetry.ids import mint_run_uid
        return "a" + mint_run_uid(payload)[1:]
    except Exception:
        blob = json.dumps(payload, sort_keys=True, default=str)
        return "a" + hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()


def initialize(args, entity, exp_name, project_name, run_spec=None):
    config_dict = namespace_to_dict(args)
    run_id = run_id_for_spec(args=args, exp_name=exp_name, run_spec=run_spec)
    if run_spec is not None or _spec_from_env() is not None:
        config_dict = dict(config_dict, run_uid=run_id)
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=entity,
        project=project_name,
        name=f"{exp_name}__{run_id}",
        config=config_dict,
        id=run_id,
        resume="allow",
    )

def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x