"""Declared invariants: turning "we assume X held" into "we observed X held".

Motivation, from this repo's own failure record
-----------------------------------------------
On 2026-08-13 nine Phase II-A runs were invalidated by the ``frozen_label_dropout``
bug: the conditioning that was *supposed* to be bitwise frozen across a finite
difference silently was not, producing ~19% effective label dropout and
destroying every FD estimate that depended on it.  The runs looked healthy.  The
loss curves looked healthy.  The defect was found by reading code, not by reading
telemetry, and it cost the campaign a full cycle of GPU time.

That bug has a shape, and the shape recurs:

* ``train.py`` asserts in a comment that "global RNG state entering this call is
  identical between arms (same ``--global-seed``, identical program flow up to
  this point)".  Nothing checks it.
* The fixed diagnostic probe is documented as "fixed and reused identically every
  step".  Nothing checks it.
* Two arms of a comparison are assumed to consume the same data in the same
  order.  Nothing checks it.

Every one of these is a *property that is cheap to hash and expensive to be wrong
about*.  This module makes them first-class telemetry: a run declares an
invariant, emits a checksum of it whenever it is convenient to do so, and the
reader verifies -- mechanically, after the fact, across runs -- that the checksum
never moved when it was supposed to be constant, and that it matched across arms
when it was supposed to match.

Two kinds of invariant
----------------------
``CONSTANT``
    A quantity that must not change over the life of one run.  Example: the
    fixed probe batch; the frozen conditioning labels; the class-embedding rows
    used in a finite difference.  Violated iff two emissions within one execution
    disagree.

``SHARED``
    A quantity that must be identical *across* runs being compared.  Example: the
    probe batch two arms are evaluated on; the data order implied by a seed; the
    resolved model architecture.  Violated iff two executions of different arms
    in the same comparison group disagree.

A ``SHARED`` violation is the one that silently invalidates an A/B result, which
is why :func:`check_group` treats it as an error rather than a warning: if the
two arms were not evaluated on the same probe, their target-cosine numbers were
never comparable, and no amount of downstream statistics repairs that.

Cost
----
Hashing a probe batch is a few hundred microseconds against a training step
measured in hundreds of milliseconds, and the emissions are O(10) per run.  The
cost of *not* doing it has already been measured, once, in nine runs.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import EventType


class InvariantKind(str, enum.Enum):
    #: Must not change within one execution.
    CONSTANT = "constant"
    #: Must be identical across every execution in a comparison group.
    SHARED = "shared"
    #: Must change (a guard against a quantity being accidentally frozen -- the
    #: dual of CONSTANT, and the check that would catch "the mining step is a
    #: no-op because the perturbation is always zero").
    VARYING = "varying"


def checksum(value: Any, *, precision: Optional[int] = None) -> str:
    """A stable 128-bit digest of a tensor, array, or plain python value.

    Floating-point hashing is a trap: bitwise-identical computations on different
    hardware, or under different autocast settings, can differ in the last ulp,
    and a checksum that flags those as violations would cry wolf until it was
    ignored.  ``precision`` rounds to a fixed number of decimals before hashing,
    which makes the digest robust to that last-bit noise while remaining
    sensitive to any change large enough to matter scientifically.  For an
    invariant that must be *bitwise* frozen -- the frozen-conditioning case that
    caused the real incident -- leave ``precision=None`` and hash the exact bytes.
    """
    hasher = hashlib.blake2b(digest_size=16)
    _feed(hasher, value, precision)
    return hasher.hexdigest()


def _feed(hasher, value: Any, precision: Optional[int]) -> None:
    # torch / numpy without importing either at module scope: this module is
    # imported by analysis code that has no reason to pay for torch.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy") and not isinstance(value, (bytes, bytearray)):
        try:
            value = value.numpy()
        except Exception:
            pass
    # Raw bytes first: memoryview also exposes `tobytes`/`shape`, so the array
    # branch below would otherwise claim it and then look for a `dtype` it
    # does not have.
    if isinstance(value, (bytes, bytearray, memoryview)):
        hasher.update(bytes(value))
        return
    if hasattr(value, "tobytes") and hasattr(value, "dtype") and hasattr(value, "shape"):
        hasher.update(str(getattr(value, "dtype", "")).encode())
        hasher.update(str(tuple(value.shape)).encode())
        if precision is None:
            hasher.update(memoryview(value.tobytes() if value.flags["C_CONTIGUOUS"]
                                     else value.copy().tobytes()))
        else:
            rounded = value.round(precision) if hasattr(value, "round") else value
            hasher.update(memoryview(rounded.copy().tobytes()))
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        hasher.update(bytes(value))
        return
    if isinstance(value, float) and precision is not None:
        hasher.update(f"{round(value, precision):.{precision}f}".encode())
        return
    if isinstance(value, Mapping):
        for key in sorted(map(str, value)):
            hasher.update(key.encode())
            _feed(hasher, value[key], precision)
        return
    if isinstance(value, (list, tuple)):
        hasher.update(f"seq{len(value)}".encode())
        for item in value:
            _feed(hasher, item, precision)
        return
    hasher.update(repr(value).encode())


@dataclasses.dataclass(frozen=True)
class Invariant:
    """A declared property, its kind, and why it matters.

    ``rationale`` is required, not decorative.  An invariant whose purpose is not
    written down gets deleted the first time it fires inconveniently; one that
    says "the FD estimate is meaningless if this moves -- see the 2026-08-13
    label-dropout incident" survives contact with a deadline.
    """

    name: str
    kind: InvariantKind
    rationale: str
    precision: Optional[int] = None


#: The invariants this codebase actually needs, derived from its incident history.
#: Registered centrally so the producer and the checker cannot drift apart.
REGISTRY: Dict[str, Invariant] = {
    inv.name: inv for inv in (
        Invariant(
            "frozen_conditioning",
            InvariantKind.CONSTANT,
            "The class conditioning consumed on both sides of a finite difference "
            "must be bitwise identical. When it was not (frozen_label_dropout, "
            "2026-08-13) every FD estimate became noise and nine Phase II-A runs "
            "were invalidated. Bitwise, hence precision=None.",
        ),
        Invariant(
            "probe_batch",
            InvariantKind.SHARED,
            "The fixed held-out diagnostic probe must be the SAME batch in every "
            "arm, or target-cosine and probe_delta_L are not comparable across "
            "arms and the central G-vs-D claim is void. train.py asserts this in "
            "a comment; this checks it.",
        ),
        Invariant(
            "probe_batch_stability",
            InvariantKind.CONSTANT,
            "The probe must also not drift WITHIN a run, or its per-step delta_L "
            "trace measures the probe changing rather than the model learning.",
        ),
        Invariant(
            "model_architecture",
            InvariantKind.SHARED,
            "Arms must instantiate the same architecture; a silent config drift "
            "between arms confounds every comparison drawn from them.",
        ),
        Invariant(
            "data_order",
            InvariantKind.SHARED,
            "Arms sharing a seed should consume the same data in the same order. "
            "train.py restores step/epoch on resume but explicitly NOT the "
            "dataloader position, so a resumed arm silently diverges here -- this "
            "invariant is how that becomes visible instead of invisible.",
            precision=None,
        ),
        Invariant(
            "fd_perturbation",
            InvariantKind.VARYING,
            "The finite-difference perturbation must actually vary; a frozen or "
            "zero perturbation makes the FD arm a no-op that still produces "
            "plausible-looking curves.",
        ),
    )
}


def emit_invariant(recorder, name: str, value: Any, *, step: Optional[int] = None,
                   **context: Any) -> Optional[str]:
    """Hash ``value`` and record it as a NOTICE on the run's event stream.

    Returns the digest so a caller can also assert on it inline.  Unknown names
    are rejected: an invariant that is not in the registry has no declared kind,
    so the checker would not know whether its changing is a violation or the
    point.
    """
    invariant = REGISTRY.get(name)
    if invariant is None:
        raise KeyError(
            f"unknown invariant {name!r}; register it in telemetry.invariants."
            "REGISTRY with a kind and a rationale before emitting it")
    digest = checksum(value, precision=invariant.precision)
    payload: Dict[str, Any] = {
        "level": "invariant",
        "message": f"invariant:{name}",
        "invariant": name,
        "invariant_kind": invariant.kind.value,
        "digest": digest,
    }
    if step is not None:
        payload["step"] = int(step)
    payload.update(context)
    # `notice` takes (message, level=..., **extra).
    recorder.notice(payload.pop("message"), level=payload.pop("level"), **payload)
    return digest


@dataclasses.dataclass
class Violation:
    invariant: str
    kind: InvariantKind
    severity: str
    detail: str
    digests: Dict[str, str] = dataclasses.field(default_factory=dict)
    rationale: str = ""


def _observations(run) -> Dict[str, List[Tuple[int, str]]]:
    """``{invariant_name: [(step, digest), ...]}`` for one execution."""
    out: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for event in run.records(EventType.NOTICE):
        name = event.get("invariant")
        if name and event.get("digest"):
            out[name].append((int(event.get("step", -1)), str(event["digest"])))
    return out


def check_run(run) -> List[Violation]:
    """Verify every CONSTANT and VARYING invariant within one execution."""
    violations: List[Violation] = []
    for name, observations in _observations(run).items():
        invariant = REGISTRY.get(name)
        if invariant is None or len(observations) < 2:
            continue
        digests = {d for _, d in observations}
        if invariant.kind is InvariantKind.CONSTANT and len(digests) > 1:
            first_change = next(
                (step for (step, d) in observations if d != observations[0][1]), None)
            violations.append(Violation(
                invariant=name, kind=invariant.kind, severity="error",
                detail=(f"changed within the run: {len(digests)} distinct digests "
                        f"over {len(observations)} observations; first change at "
                        f"step {first_change}"),
                digests={str(step): d for step, d in observations[:6]},
                rationale=invariant.rationale))
        if invariant.kind is InvariantKind.VARYING and len(digests) == 1:
            violations.append(Violation(
                invariant=name, kind=invariant.kind, severity="error",
                detail=(f"never changed across {len(observations)} observations; "
                        "the quantity that was supposed to vary is frozen"),
                digests={"all": observations[0][1]},
                rationale=invariant.rationale))
    return violations


def check_group(runs: Sequence[Any], labels: Optional[Sequence[str]] = None
                ) -> List[Violation]:
    """Verify SHARED invariants across the executions of a comparison group.

    This is the check that decides whether an A/B result is admissible at all.
    A SHARED violation is an ERROR, not a warning: if two arms were measured on
    different probes, no downstream statistic makes them comparable again.
    """
    labels = list(labels) if labels else [getattr(r, "exec_id", str(i))
                                          for i, r in enumerate(runs)]
    per_run = [(_observations(run), label) for run, label in zip(runs, labels)]
    violations: List[Violation] = []
    names = {n for obs, _ in per_run for n in obs}
    for name in sorted(names):
        invariant = REGISTRY.get(name)
        if invariant is None or invariant.kind is not InvariantKind.SHARED:
            continue
        seen: Dict[str, str] = {}
        for observations, label in per_run:
            values = observations.get(name)
            if values:
                seen[label] = values[0][1]
        if len(set(seen.values())) > 1:
            violations.append(Violation(
                invariant=name, kind=invariant.kind, severity="error",
                detail=("differs across arms: "
                        + ", ".join(f"{k}={v[:12]}" for k, v in sorted(seen.items()))
                        + " -- these runs were not measured on the same quantity, "
                          "so comparing them is invalid"),
                digests=seen, rationale=invariant.rationale))
        missing = [label for _, label in per_run if label not in seen]
        if seen and missing:
            violations.append(Violation(
                invariant=name, kind=invariant.kind, severity="warn",
                detail=(f"not emitted by {', '.join(missing)}, so the shared "
                        "property is unverified for those arms"),
                digests=seen, rationale=invariant.rationale))
    return violations


def render(violations: Iterable[Violation]) -> str:
    violations = list(violations)
    if not violations:
        return "invariants: all declared invariants held.\n"
    lines = [f"invariants: {len(violations)} violation(s)\n"]
    for violation in sorted(violations, key=lambda v: (v.severity != "error", v.invariant)):
        lines.append(f"  [{violation.severity.upper()}] {violation.invariant} "
                     f"({violation.kind.value}): {violation.detail}")
        if violation.rationale:
            lines.append(f"      why it matters: {violation.rationale}")
    return "\n".join(lines) + "\n"
