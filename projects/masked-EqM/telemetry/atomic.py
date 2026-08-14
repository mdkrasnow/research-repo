"""Crash-atomic file publication: a path either has its old contents or its new
contents, never a prefix of the new contents.

The defect this replaces
------------------------
``torch.save(obj, "/.../checkpoints/0125000.pt")`` writes *in place*.  The name
``0125000.pt`` becomes visible, and claims to be a step-125000 checkpoint, from
the instant the file is created -- which is before a single tensor has been
written.  If the process dies in the window between ``open`` and the final
``close`` (SLURM preemption, node failure, walltime kill, quota EDQUOT), what
survives is a truncated file at a name that every downstream consumer treats as
a valid checkpoint.  ``torch.load`` then fails with a zip/unpickle error far
from the cause, or -- worse, for formats without a trailing checksum -- succeeds
on a partial payload.

The invariant we restore
------------------------
Let ``P`` be the canonical path and ``C(P, t)`` its contents at time ``t``.  The
publication is *linearizable* if for every ``t`` either ``C(P, t) = old`` or
``C(P, t) = new``.  POSIX ``rename(2)`` within a single filesystem is specified
to be atomic with respect to other processes: "If the link named by the new
argument exists, it shall be removed and old renamed to new... this rename shall
be atomic relative to other threads/processes accessing the file."  So writing
the full payload to a *different* name and then ``rename``-ing it over ``P``
gives exactly that two-state property, with the crash-visible intermediate state
being a temp file under a name no consumer reads.

Two conditions are load-bearing and are the reason this helper exists rather
than three-line copies at each call site:

1. **The temp file must live in the same directory as the target.**  ``rename``
   across filesystems fails with ``EXDEV``; a temp in ``/tmp`` with a target on
   ``netscratch`` is not a rename at all.  The existing copies in the repo get
   this right by construction (``dir=path.parent``); a future copy-paste into a
   ``tempfile.mkstemp()`` default would not.

2. **The payload must be durable before the rename.**  ``rename`` orders only
   the *metadata* operation.  Without an ``fsync`` on the file descriptor, a
   power loss or node crash can leave the directory entry pointing at a file
   whose data blocks were never flushed -- the rename is atomic but publishes
   zeroes.  Neither existing copy in this repo (``masked_field_shaping/
   train_continuation.py``, ``energy_monotonicity/evaluate_energy_monotonicity.py``)
   fsyncs, so both are atomic against *process* death but not against *node*
   death.  This module fsyncs the file and then the containing directory (the
   latter makes the new directory entry itself durable).

On network filesystems (Lustre/NFS, i.e. netscratch and holylabs) same-directory
rename atomicity holds; cross-client cache coherence is a separate concern and is
not what this helper claims to solve.

Step/payload agreement
----------------------
A checkpoint encodes its step twice: in the filename and in the payload.  These
can disagree if a save is retried with a different step, or if a file is copied
under a new name.  :func:`atomic_torch_save` therefore accepts an optional
``expect_step_key``; when given, it asserts the payload's step field equals the
step parsed from the filename *before* publishing, turning a silent provenance
corruption into a loud failure at the only moment it is cheap to catch.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Union

__all__ = [
    "atomic_write_bytes",
    "atomic_torch_save",
    "atomic_json_dump",
    "step_from_filename",
    "AtomicWriteError",
]

PathLike = Union[str, "os.PathLike[str]", Path]


class AtomicWriteError(RuntimeError):
    """Raised when a payload fails its pre-publication consistency check."""


_STEP_RE = re.compile(r"(\d+)")


def step_from_filename(path: PathLike) -> Optional[int]:
    """Parse the step encoded in a checkpoint filename, or ``None``.

    Recognizes the two conventions in this repo: ``0125000.pt`` (zero-padded
    step) and ``epoch80.pt`` (epoch-tagged, which is *not* a step and so returns
    ``None`` -- an epoch tag must not be compared against a step field).
    """
    stem = Path(path).stem
    if not stem or not stem.isdigit():
        return None
    return int(stem)


def _fsync_dir(directory: Path) -> None:
    """Make a directory entry durable.

    Required after ``os.replace``: the rename is atomic, but the *directory*
    block recording it is not necessarily on stable storage until the directory
    itself is synced.  Best-effort -- some filesystems reject ``O_RDONLY`` fsync
    on directories, and failing the whole save over that would be worse than the
    residual risk.
    """
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: PathLike, writer: Callable[[Any], None]) -> Path:
    """Publish ``path`` atomically; ``writer`` receives an open binary file object.

    The generic core: every other function here is a thin specialization.  The
    sequence is write -> flush -> fsync -> close -> os.replace -> fsync(dir),
    and on any exception the temp file is removed so a failed save leaves no
    litter for the checkpoint pruner to trip over.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # delete=False because we hand the name to os.replace; the finally-block
    # below is what guarantees no orphan survives a failure.
    handle = tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)
    return path


def atomic_torch_save(obj: Any, path: PathLike, *,
                      expect_step_key: Optional[str] = None,
                      **torch_save_kwargs: Any) -> Path:
    """``torch.save`` with crash-atomic publication.

    Argument order matches ``torch.save(obj, path)`` deliberately, so adopting it
    at a call site is a pure identifier substitution with no argument reordering
    -- the kind of edit that cannot introduce a transposition bug.

    ``expect_step_key``: if given and the filename encodes a step, assert
    ``obj[expect_step_key] == step_from_filename(path)`` before publishing, so a
    checkpoint whose name and payload disagree is never created at all.
    """
    import torch  # imported lazily: consumers of atomic_json_dump need no torch

    if expect_step_key is not None:
        expected = step_from_filename(path)
        if expected is not None:
            try:
                actual = obj[expect_step_key]
            except (TypeError, KeyError):
                actual = None
            if actual is not None and int(actual) != expected:
                raise AtomicWriteError(
                    f"refusing to write {Path(path).name}: filename encodes step "
                    f"{expected} but payload[{expect_step_key!r}]={actual}. The "
                    "name and the payload must agree or provenance is unrecoverable."
                )

    return atomic_write_bytes(path, lambda fh: torch.save(obj, fh, **torch_save_kwargs))


def atomic_json_dump(obj: Any, path: PathLike, *, indent: int = 2,
                     sort_keys: bool = True, default: Optional[Callable] = None) -> Path:
    """Serialize ``obj`` as JSON and publish it atomically.

    ``sort_keys=True`` by default: a JSON artifact that is byte-stable under
    re-derivation is diffable and content-hashable, which is what makes it usable
    as provenance rather than merely as output.
    """
    def _write(fh):
        text = json.dumps(obj, indent=indent, sort_keys=sort_keys,
                          default=default or _json_default)
        fh.write((text + "\n").encode("utf-8"))

    return atomic_write_bytes(path, _write)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):          # numpy scalars/arrays, torch tensors
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)
