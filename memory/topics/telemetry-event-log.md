---
name: telemetry-event-log
description: The masked-EqM telemetry rewrite — content-addressed run identity, guaranteed terminal records, and completeness-gated aggregation. Replaces path-parsed identity and run-relative analysis windows.
status: active
---

# Telemetry event log (`projects/masked-EqM/telemetry/`)

Built 2026-08-14. Origin: user reported "workflow start emissions are not directly tied to
workflow finish emissions … a lot of bulk telemetry, but not much wiring between telemetry."

## The one-sentence diagnosis

There was **no primary key on the run entity**. Four artifacts claimed to describe runs and
no two shared a key; the stream the science was computed from (`gradient_metrics.jsonl`)
carried no identity at all, so consumers recovered it by parsing filesystem paths — a lossy,
non-injective decoding. And there were **no finish emissions at all**: `train.py` closed its
metrics file without writing anything, so a run killed at step 3k and a run that completed
20k left identically-shaped telemetry.

## Governing principle

**The append-only event log is the only source of truth; every state document is a fold over
it and must be fully regenerable.** A mutable `status` field that a writer overwrites
destroys history and can silently disagree with reality — exactly what `results/btm/
manifest.jsonl` did by freezing `status: "submitted"` forever on 12 rows that were all
actually cancelled.

## The three concepts, kept strictly separate

- `run_uid` — identity of a LOGICAL experiment. blake2b-64 content hash of the canonicalized
  spec. Deterministic, so resubmitting an identical spec yields an identical uid and the two
  are joinable by construction. Also makes spec drift *detectable* (`verify_spec`) and makes
  "is this comparison controlled?" computable (`differing_fields`).
- `exec_id` = `<run_uid>:<job_id>:a<attempt>` — identity of one PHYSICAL execution.
  **A SLURM requeue reuses the job id**, so job_id alone is not a key; hence the attempt
  counter, derived from what is on disk rather than trusted from the environment.
  Invalidation ("that run was contaminated") applies to executions, not logical runs.
- `seq` — monotone per-execution counter. Makes record loss *provable* (a gap is proof;
  a timestamp-ordered log can never establish this) rather than merely suspected.

## The five-layer terminal-record ladder

Every execution that emits START also emits exactly one END:
1. normal return → `__exit__`
2. exception → same, status `crashed`, with traceback
3. signal (SIGTERM/SIGUSR1/SIGINT) → handler raises `Interrupted(BaseException)`. It derives
   from BaseException *specifically* so the broad `except Exception` handlers in training
   loops cannot swallow a preemption notice
4. `atexit`
5. SIGKILL / node death → unreachable from inside the process by definition. Covered by the
   shell sealer (`telemetry/seal.py`, invoked from the sbatch trap) and, failing that, the
   reconciler inferring from `sacct` and appending END with `inferred: true`.

`RunStatus.LOST` exists as a distinct status because "we know it died" and "we never found
out" are different epistemic states.

## The scientific gate (`read.py`)

- **Absolute windows** on the PLANNED step axis, shared across arms. `shared_windows()`
  REFUSES to build a common axis for runs planned to different lengths rather than silently
  reconciling them.
- **`CompletenessPolicy`** — a total predicate with an explicit reason per rejection.
  Quarantined runs are returned *alongside* the admitted ones so a report cannot lie by
  omission: how many runs died is part of the result.
- `seq` gaps quarantine a run even if it has a clean END — its statistics are over an
  unknown subsample.

## Declared invariants (`invariants.py`) — the novel piece

Aimed at the bug class that already cost this campaign 9 runs (`frozen_label_dropout`,
2026-08-13, ~19% effective label dropout across a finite difference). A run declares a
property, emits a cheap checksum, and the reader verifies CONSTANT ones never moved within a
run and SHARED ones matched across arms. Registry entries carry a required `rationale` — an
invariant whose purpose isn't written down gets deleted the first time it fires
inconveniently.

Rationale for existing: `train.py` *asserts in a comment* that "global RNG state entering
this call is identical between arms" and that the probe is "fixed and reused identically
every step". Nothing checked either. Each is cheap to hash and expensive to be wrong about.

## Gotcha worth remembering

`RunRecorder` first sealed every *healthy* run as `timeout`: `last_step` is the last
**logged** step, and at a logging cadence of 50 a run that executed all 20,000 steps reports
19,950. Fixed with `mark_complete()` (the loop is the only thing that knows it finished)
plus a stride-aware fallback measuring the observed cadence. **Completion is an assertion,
not an inference from a step counter.**

## Related

- [[btm-fd-scalar-campaign]] — the campaign whose Phase II-A analysis this protects.
- Audit: `projects/masked-EqM/documentation/telemetry-audit-2026-08-14.md`
- Daily log: `memory/2026-08-14.md`
