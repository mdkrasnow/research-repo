"""Mint an ``EQM_RUN_SPEC`` blob from the command line.

Why a CLI instead of building the JSON in bash
----------------------------------------------
``RunSpec.from_env()`` reconstructs a spec by ``json.loads``-ing whatever the
launcher put in ``EQM_RUN_SPEC``, and the ``run_uid`` it yields is a content hash
of that structure.  Hand-assembling the blob with ``printf`` in a shell script
would therefore make the *identity of the experiment* a function of bash quoting:
an unescaped path, a locale-dependent float rendering, or a key typed in a
different order in one launcher than another would silently produce a different
``run_uid`` for the same experiment -- reintroducing exactly the "two names for
one thing" defect that content addressing exists to remove.

So the shell states *fields*, and this module -- which imports the same
:class:`telemetry.ids.RunSpec` the trainer will import -- produces the bytes.
There is one serializer, so the launcher's view and the trainer's view of the
spec cannot drift.

Usage from the prelude::

    EQM_RUN_SPEC="$(python -m telemetry.mkspec \
        --campaign btm --phase II --arm btm_scalar_fd_directional \
        --seed 0 --git-sha "$GIT_SHA" --planned-steps 20000 \
        --param fd_k=4 --param fd_eps=1e-3 --param ebm=direct)"

Parameter typing
----------------
``--param k=v`` values are parsed as JSON when that succeeds and kept as strings
otherwise.  This is deliberate and load-bearing: ``fd_k=4`` must hash as the
integer 4 and not the string ``"4"``, because the trainer will pass an integer
and :func:`telemetry.ids.canonicalize` distinguishes the two.  ``1e-3`` becomes a
float, which ``canonicalize`` then normalizes through ``repr`` so that ``1e-3``
and ``0.001`` agree.  Anything not valid JSON (a filesystem path, an arm name)
stays a string, which is what it was.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # pragma: no cover - direct-script path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from telemetry.ids import RunSpec  # type: ignore
else:
    from .ids import RunSpec


def parse_param(item: str) -> tuple:
    """``"k=v"`` -> ``(k, typed_v)``; see the module docstring on typing."""
    if "=" not in item:
        raise argparse.ArgumentTypeError(
            f"--param expects KEY=VALUE, got {item!r}")
    key, _, raw = item.partition("=")
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"--param has an empty key: {item!r}")
    raw = raw.strip()
    if raw == "":
        # An empty value is "the launcher had nothing to say about this key".
        # Emitting None lets canonicalize() drop it, so an unset optional knob
        # hashes identically to one that was never mentioned -- which is the
        # behavior that keeps two launchers for the same arm agreeing.
        return key, None
    try:
        return key, json.loads(raw)
    except ValueError:
        return key, raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m telemetry.mkspec",
        description="Emit the canonical EQM_RUN_SPEC JSON blob on stdout.")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--phase", default="")
    parser.add_argument("--planned-steps", default=None, type=int)
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="Identity-bearing parameter; repeatable.")
    parser.add_argument("--print-uid", action="store_true",
                        help="Print only the run_uid (for log lines and dir names).")
    parser.add_argument("--print-slug", action="store_true",
                        help="Print only the slug (for human-facing directory names).")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    params: Dict[str, Any] = {}
    for item in args.param:
        key, value = parse_param(item)
        if key in params and params[key] != value:
            # Repeating a key with two different values means the launcher
            # contradicted itself; silently taking the last one would bake an
            # arbitrary choice into the run's identity.
            raise SystemExit(
                f"[telemetry.mkspec] --param {key} given twice with different "
                f"values ({params[key]!r} then {value!r})")
        params[key] = value
    spec = RunSpec(
        campaign=args.campaign, arm=args.arm, seed=args.seed,
        git_sha=args.git_sha, phase=args.phase,
        planned_steps=args.planned_steps, params=params,
    )
    if args.print_uid:
        sys.stdout.write(spec.run_uid)
    elif args.print_slug:
        sys.stdout.write(spec.slug())
    else:
        sys.stdout.write(spec.to_env())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
