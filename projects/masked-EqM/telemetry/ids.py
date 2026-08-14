"""Run identity: content-addressed logical runs, and executions beneath them.

The defect this replaces
------------------------
Identity used to be recovered by *decoding a filesystem path*::

    tag = os.path.basename(os.path.dirname(os.path.dirname(p)))   # analyze_image.py
    for k in ARM_OF:
        if k in p or k in tag:
            arm = k

That decoding is neither injective nor total.  Not injective: the arm names
``btm_scalar_fd_directional`` and ``btm_scalar_fd_directional4`` are prefixes of
one another, so K=1 and K=4 runs decoded to the *same* arm and were averaged
together.  Not total: an unrecognized tag yields the sentinel ``"?"`` which was
still aggregated.  A scheme in which identity must be *parsed* will keep
producing bugs of this shape indefinitely, because the parser and the namer live
in different files and drift apart.

The fix is to make identity something the producer *states* and the consumer
*reads*, never derives.  Concretely: a content hash over the canonicalized
specification.

Why content-addressed rather than a random UUID
-----------------------------------------------
A UUID would also solve the join problem, but a content hash additionally gives:

* **Idempotent identity.** Re-submitting the same spec (after a preemption, or
  from a different machine) yields the same ``run_uid``, so the two executions
  are automatically recognized as attempts at the same logical run instead of
  looking like two unrelated runs that happen to be similar.
* **Spec-drift detection.** If anyone edits a launcher default, the hash moves.
  A run whose ``run_uid`` does not match the hash of its own recorded spec is
  proof that the spec was mutated after minting -- a check `verify_spec` makes
  cheap and that `telemetry doctor` runs routinely.
* **Comparability by construction.** Two runs share every spec field except the
  one under test iff their specs differ in exactly that field, which is
  mechanically checkable (`differing_fields`) rather than asserted in prose.

The hash covers ONLY fields that change what is computed.  Wall-clock, partition,
number of dataloader workers and output paths are deliberately excluded: two
executions that differ only in which partition they landed on are the same
experiment, and must hash the same or the whole idempotence property collapses.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from .schema import SchemaError, validate_exec_id, validate_run_uid

#: Fields excluded from the identity hash: they affect *where* and *how fast* a
#: run happens, never *what it computes*.  Keeping this list explicit (rather
#: than hashing a curated subset) means a newly added scientific parameter is
#: included by default -- fail-safe in the direction of "distinct specs get
#: distinct ids", which is the direction that preserves correctness.
NON_IDENTIFYING_KEYS = frozenset({
    "partition", "n_gpus", "num_workers", "results_dir", "results_root",
    "expected_runtime", "submitted_at", "command", "job_id", "attempt",
    "wandb", "log_every", "grad_log_every", "ckpt_every", "save_epochs",
    "description", "gate", "sbatch_path", "time_limit", "mem", "cpus_per_task",
})

_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def canonicalize(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Reduce a spec to its identity-bearing, order-independent form.

    Canonicalization rules, each closing a way two *identical* experiments could
    otherwise hash differently:

    * non-identifying keys dropped (see :data:`NON_IDENTIFYING_KEYS`);
    * ``None`` values dropped, so "parameter absent" and "parameter explicitly
      null" agree -- launchers routinely pass ``max_steps=None`` for "no limit";
    * floats normalized via ``repr`` of the float, so ``1e-3`` and ``0.001``
      agree;
    * booleans normalized ahead of ints, since ``bool`` is an ``int`` subclass
      and ``True``/``1`` would otherwise collide;
    * keys sorted by ``json.dumps(sort_keys=True)``.
    """
    out: Dict[str, Any] = {}
    for key, value in spec.items():
        key = str(key)
        if key in NON_IDENTIFYING_KEYS or value is None:
            continue
        if isinstance(value, bool):
            out[key] = bool(value)
        elif isinstance(value, float):
            out[key] = repr(float(value))
        elif isinstance(value, (int, str)):
            out[key] = value
        elif isinstance(value, Mapping):
            nested = canonicalize(value)
            if nested:
                out[key] = nested
        elif isinstance(value, (list, tuple)):
            out[key] = [canonicalize(v) if isinstance(v, Mapping)
                        else (repr(float(v)) if isinstance(v, float) and not isinstance(v, bool) else v)
                        for v in value]
        else:
            out[key] = str(value)
    return out


def mint_run_uid(spec: Mapping[str, Any]) -> str:
    """Content-address a specification into a ``run_uid``.

    BLAKE2b truncated to 64 bits.  The birthday bound puts the collision
    probability below 1e-9 until roughly 2^17 distinct specs; this campaign has
    produced O(10^2).  Truncating keeps the id short enough to appear in
    directory names and log lines without hurting readability, which matters
    because an id nobody is willing to type is an id that gets replaced by a
    nickname -- and nicknames are what caused the three-vocabulary problem.
    """
    canonical = json.dumps(canonicalize(spec), sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()
    return f"r{digest}"


def slug_job_id(job_id: Any) -> str:
    """Normalize a scheduler job id into the exec_id's middle field.

    Exposed so that post-mortem tooling (the shell sealer, the reconciler) can
    locate an execution's stream without re-deriving the id format by splitting
    a synthesized exec_id. Two files independently knowing how ids are spelled is
    how they drift apart.
    """
    return _SLUG_RE.sub("-", str(job_id) if job_id not in (None, "") else "local")


def make_exec_id(run_uid: str, job_id: Any, attempt: int = 0) -> str:
    """Identity of one physical execution: ``<run_uid>:<job_id>:a<attempt>``.

    ``job_id`` is slugified because SLURM array ids contain ``_`` and local runs
    have no id at all (we use ``local``), and because a separator collision would
    make the id ambiguous to split.

    ``attempt`` distinguishes executions that share a job id: a SLURM *requeue*
    reuses the job id, so job id alone is not a key for an execution.  Two
    requeued attempts writing to the same stream under the same id would produce
    a non-monotone ``seq`` sequence and interleaved step numbers -- silently, and
    exactly the way the old append-mode metrics file did.
    """
    validate_run_uid(run_uid)
    slug = slug_job_id(job_id)
    if not isinstance(attempt, int) or attempt < 0:
        raise SchemaError(f"attempt must be a non-negative int, got {attempt!r}")
    return validate_exec_id(f"{run_uid}:{slug}:a{attempt}")


def split_exec_id(exec_id: str) -> Tuple[str, str, int]:
    """Inverse of :func:`make_exec_id` -> ``(run_uid, job_id, attempt)``."""
    validate_exec_id(exec_id)
    run_uid, job_id, attempt = exec_id.split(":")
    return run_uid, job_id, int(attempt[1:])


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """The immutable, identity-bearing description of one logical run.

    ``params`` holds everything arm-specific (btm mode, fd_k, fd_eps, tc, lr,
    batch size, interpolant...).  It is deliberately open rather than a fixed
    field list: a closed schema would have to be edited for every new
    experiment, and the edit would be forgotten, and the forgotten parameter
    would silently not participate in identity.
    """

    campaign: str
    arm: str
    seed: int
    git_sha: str
    phase: str = ""
    planned_steps: Optional[int] = None
    params: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def identity(self) -> Dict[str, Any]:
        """The exact mapping that gets hashed."""
        return canonicalize({
            "campaign": self.campaign,
            "arm": self.arm,
            "seed": int(self.seed),
            "git_sha": self.git_sha,
            "phase": self.phase,
            "planned_steps": self.planned_steps,
            "params": dict(self.params),
        })

    @property
    def run_uid(self) -> str:
        return mint_run_uid(self.identity())

    def slug(self) -> str:
        """Human-facing name.  Never a key -- only ever decoration.

        Every historical identity bug in this repo came from a human-readable
        name being pressed into service as a join key.  The slug embeds the
        ``run_uid`` so that a directory listing remains greppable by the real
        key, and so that a human who copies a directory name into a bug report
        has copied something machine-resolvable.
        """
        parts = [p for p in (self.campaign, self.phase, self.arm, f"s{self.seed}") if p]
        stem = _SLUG_RE.sub("-", "_".join(parts))
        return f"{stem}__{self.run_uid}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign": self.campaign, "phase": self.phase, "arm": self.arm,
            "seed": int(self.seed), "git_sha": self.git_sha,
            "planned_steps": self.planned_steps, "params": dict(self.params),
            "run_uid": self.run_uid, "slug": self.slug(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunSpec":
        return cls(
            campaign=data["campaign"], arm=data["arm"], seed=int(data["seed"]),
            git_sha=data["git_sha"], phase=data.get("phase", ""),
            planned_steps=data.get("planned_steps"),
            params=dict(data.get("params") or {}),
        )

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "RunSpec":
        """Reconstruct the spec a launcher passed through the environment.

        The launcher exports ``EQM_RUN_SPEC`` (canonical JSON).  Reconstructing
        from a single serialized blob -- rather than from a dozen individual
        env vars re-read by the trainer -- is what keeps the trainer's view of
        the spec bit-identical to the launcher's.  Re-deriving the spec on the
        far side of the shell is precisely how a launcher default and a trainer
        default drift apart without anyone noticing.
        """
        env = os.environ if env is None else env
        blob = env.get("EQM_RUN_SPEC")
        if not blob:
            raise SchemaError(
                "EQM_RUN_SPEC is not set: this process was not launched through "
                "the telemetry-aware launcher, so its run identity is unknown. "
                "Refusing to guess.")
        return cls.from_dict(json.loads(blob))

    def to_env(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def verify_spec(run_uid: str, spec: Mapping[str, Any]) -> None:
    """Assert that ``run_uid`` really is the hash of ``spec``.

    Cheap, and it catches the class of failure where a run's recorded spec was
    edited after the fact -- e.g. a results table hand-corrected to say ``fd_k:
    4`` on a run that actually executed ``fd_k: 1``.  Under content addressing
    that edit is *detectable*, which is the entire practical payoff of hashing
    over random ids.
    """
    expected = mint_run_uid(spec)
    if expected != run_uid:
        raise SchemaError(
            f"run_uid {run_uid} does not match the hash of its own spec "
            f"({expected}): the specification was mutated after minting")


def differing_fields(a: RunSpec, b: RunSpec) -> Dict[str, Tuple[Any, Any]]:
    """Every identity-bearing field on which two runs differ.

    Used by the analyzer to *prove* that a comparison is controlled: an A/B
    claim is only sound if the two arms' specs differ in exactly the field under
    test.  Previously this was asserted in prose in a launcher docstring; here
    it is a computable predicate the report can print.
    """
    flat_a, flat_b = _flatten(a.identity()), _flatten(b.identity())
    keys = set(flat_a) | set(flat_b)
    return {k: (flat_a.get(k), flat_b.get(k)) for k in sorted(keys)
            if flat_a.get(k) != flat_b.get(k)}


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out
