"""masked-EqM run telemetry: identity, lifecycle, and analyzable event logs.

Design in one paragraph
-----------------------
Every physical execution appends typed, sequenced events to its own file.  The
identity of the experiment is a content hash of its specification, stated by the
producer and never re-derived by a consumer from a filesystem path.  Every
execution that starts also ends, guaranteed by a five-layer ladder that reaches
from a normal return down to a node dying without warning.  All state documents
(ledgers, status tables, results TSVs) are folds over the event log and are
regenerable from it; none of them is authoritative.  Aggregation is gated on a
machine-checkable completeness predicate, so a truncated run cannot silently be
pooled with a complete one.

Quick start (producer)::

    from telemetry import RunSpec, RunRecorder, recorder_for_rank

    spec = RunSpec.from_env()                    # launcher exported EQM_RUN_SPEC
    with recorder_for_rank(rank, root, spec, planned_steps=20_000) as run:
        for step in range(...):
            run.progress(step, kind="grad", grad_norm=g, clipped=c)
            run.heartbeat(step)
        run.set_last_step(step)

Quick start (consumer)::

    from telemetry import load_campaign

    campaign = load_campaign(root)
    usable = campaign.analyzable()               # complete runs only
    for run in campaign.quarantined():
        print(run.exec_id, run.rejection_reason)
"""

from .schema import (  # noqa: F401
    SCHEMA_VERSION,
    EventType,
    RunStatus,
    SchemaError,
    TERMINAL_STATUSES,
    TRUNCATED_STATUSES,
    dumps,
    make_record,
    validate_record,
)
from .ids import (  # noqa: F401
    RunSpec,
    differing_fields,
    make_exec_id,
    mint_run_uid,
    split_exec_id,
    verify_spec,
)
from .emit import (  # noqa: F401
    JsonlSink,
    StderrSink,
    TelemetryWriter,
    WandbSink,
    open_writer,
)
from .lifecycle import (  # noqa: F401
    Interrupted,
    NullRecorder,
    RunRecorder,
    next_attempt,
    recorder_for_rank,
)

__all__ = [
    "SCHEMA_VERSION", "EventType", "RunStatus", "SchemaError",
    "TERMINAL_STATUSES", "TRUNCATED_STATUSES", "dumps", "make_record",
    "validate_record", "RunSpec", "differing_fields", "make_exec_id",
    "mint_run_uid", "split_exec_id", "verify_spec", "JsonlSink", "StderrSink",
    "TelemetryWriter", "WandbSink", "open_writer", "Interrupted",
    "NullRecorder", "RunRecorder", "next_attempt", "recorder_for_rank",
]


def __getattr__(name):
    """Lazily expose the read side.

    ``telemetry.read`` pulls in the aggregation machinery, which a training job
    has no use for.  Keeping it out of the eager import path means the producer
    side stays cheap to import inside a hot training process.
    """
    if name in ("load_campaign", "RunLog", "Campaign", "load_run_dir",
                "CompletenessPolicy", "absolute_windows"):
        from . import read
        return getattr(read, name)
    if name in ("build_ledger", "fold", "LedgerView"):
        from . import ledger
        return getattr(ledger, name)
    if name in ("reconcile", "classify", "Disagreement", "file_transaction"):
        from . import reconcile as _reconcile
        return getattr(_reconcile, name)
    if name in ("pipeline_view", "results_rows", "render_tsv"):
        from . import views
        return getattr(views, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
