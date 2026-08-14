# Telemetry audit — masked-EqM run instrumentation (2026-08-14)

Scope: the emission → storage → aggregation path for cluster training runs, as
exercised by the BTM/FD-scalar campaign. Files audited:

- `experiments/btm/launch_image_phase.py` (submit-side emitter)
- `slurm/jobs/btm_image_arm.sbatch` (execution wrapper)
- `train.py:336-360, 600-745, 811` (in-run emitter)
- `experiments/btm/analyze_image.py` (consumer)
- `.state/pipeline.json`, `scripts/cluster/reconcile_pipeline.py` (ledger)
- `experiments/direct_energy/campaign.py` (prior, better-wired pattern)

---

## 1. The core defect, stated precisely

The system emits four independent event streams:

| Stream | Written by | Identity carried |
|---|---|---|
| `results/btm/manifest.jsonl` | launcher, at submit | `run_tag`, `job_id`, `git_sha` |
| `.state/pipeline.json:active_runs` | hand-edited + reconciler | `run_id`, `job_id` |
| `<RESULTS_DIR>/<NNN-...>/gradient_metrics.jsonl` | `train.py`, rank 0 | **none** |
| `slurm/logs/<RUN_TAG>_<JOBID>.log` | sbatch tee | filename only |

**No two of these share a key.** There is no primary key on the *run* entity and
therefore no referential integrity between the record that says a run began and
the records that say what it did. `gradient_metrics.jsonl` — the only stream the
scientific conclusions are computed from — carries no run identity at all.

The consumer compensates by *decoding identity out of the filesystem path*
(`analyze_image.py:125-143`): it takes `basename(dirname(dirname(path)))`, which
happens to be `${RUN_TAG}_job${SLURM_JOB_ID}`, and recovers the arm by substring
matching against a hand-maintained alias table. This decoding function is not
injective (§2.2) and not total (`arm = "?"` is a reachable state that still gets
aggregated). Identity recovered by string parsing is not identity; it is a guess
that happens to have been right so far.

The user-observed symptom — "workflow start emissions are not tied to workflow
finish emissions" — is the visible surface of this. But the deeper problem is
that **there are no finish emissions at all** (§2.1), so the tie could not be
made even if the key existed.

Invariant the system needs and does not have:

> For every run `r`: exactly one START record, zero or more PROGRESS records, and
> exactly one END record, all carrying the same immutable `run_uid`; and no
> aggregation may read a run that lacks a terminal record.

---

## 2. Defects, by severity

### S1 — No terminal event; the analyzer silently compares non-comparable windows

`train.py:811` closes `grad_metrics_file` without writing anything. A run that
died at step 3,000 (timeout, preempt, NCCL failure, OOM) and a run that finished
all 20,000 steps produce byte-identical *shapes* of telemetry — a stream that
just stops.

`analyze_image.py:53-68` then defines windows **relative to each run's own
observed step range**: `lo, hi = rows[0]["step"], rows[-1]["step"]`. The headline
table (`analyze_image.py:182-198`, "Key mechanistic comparison — clip rate and
target cosine, late window") groups by `window == "late"` across runs.

Consequence: if arm G dies at step 3k and arm D runs to 20k, the table compares
G's steps 2k–3k against D's steps 13k–20k and prints them side by side as "late",
with no indication. The campaign's central hypothesis is *specifically about late-
training behaviour* — the original direct-scalar failure "was invisible in the
loss and only showed up as a growing clip rate late in training" (the module's own
docstring). This defect can invert the campaign's conclusion, and it is invisible
in the output.

Note the interaction with real cluster behaviour: these are 24h `seas_gpu` jobs
and the campaign has already lost runs to preemption and contamination. This is
not hypothetical.

### S1 — The manifest is a dead ledger describing only invalidated runs

`launch_image_phase.py:118-127` writes `"status": "submitted"` at submit time.
Nothing anywhere ever updates that field — the launcher is the only writer of
`manifest.jsonl` in the repo. `status` is therefore not a status; it is a constant.

Current contents: 12 records, job ids `39067773`–`39067818`, all
`"status": "submitted"`. Per `pipeline.json` and the 2026-08-13 commits, **every
one of those runs was invalidated** by the label-dropout contamination. The three
clean reruns the conclusions will actually rest on (`39090540/41/42`) were
submitted by hand and **appear in the manifest zero times**.

The file the launcher docstring calls "the machine-readable experiment manifest
the campaign is required to keep" currently describes exclusively runs that must
not be cited, and omits every run that must be.

### S2 — Two record types multiplexed on one stream with no discriminator

`train.py:708` writes the gradient record; `train.py:743` writes a *different*
record shape (`checklist_record`: `delta_theta_norm`, `probe_*`, `P_t`,
`eta_func`) to the **same file, at the same step, with no `type` field**.

Downstream (`analyze_image.py:48`) every line with a `step` key is treated as one
observation. Consequences:

- `n` (reported as the window's sample count) is inflated ~2×.
- `update_over_param` (`analyze_image.py:101`) divides a median taken over
  checklist rows by a median taken over gradient rows — a ratio of statistics
  from two disjoint populations. (It is currently NaN because `param_norm` is
  never logged, which masks the bug rather than fixing it.)
- Per-step joins (e.g. "was this step clipped, and what did it do to the probe
  loss?") are impossible, though both facts are recorded at the same step.

Related: the two records use the same `step` value but straddle `opt.step()`
(line 708 is pre-step, 743 is post-step). Same label, different semantics.

### S2 — Alias decoding collapses K=1 and K=4 into one arm

`analyze_image.py:128-141`. Arm resolution runs the long-form `ARM_OF` keys
first; both `btm_scalar_fd_directional` and `btm_scalar_fd_directional4` contain
the substring `btm_scalar_fd_directional`, so both map to the same arm. The `K`
suffix that would distinguish them is derived **only** from the short aliases
`_D1_`/`_D4_`, which the long-form tags do not contain.

So for the 12 manifest runs (long-form tags), K=1 and K=4 receive the identical
label `"D  FD directional"` and are **averaged together** in the final table.
FD estimator variance as a function of K is one of the campaign's decision axes
(Outcome C). It currently works only because the reruns happened to use short
tags — a naming accident, not a guarantee.

Also in this block: the long-form loop has no `break`, so on a tag matching
multiple keys the *last* dict-order match wins arbitrarily; `arm = "?"` is never
rejected; and `seed` is parsed as `tag.rsplit("_s", 1)[-1][:1]`, a single
character (seed ≥ 10 silently truncates to its first digit).

### S2 — Three incompatible identifier vocabularies for the same run

Same run, three names:

- manifest: `btm_IIA_btm_scalar_exact_s0`
- `pipeline.json`: `btm_IIA_G_s0`
- analyzer label: `G  scalar exact`

None is derivable from another without a hand-maintained table, and the tables
live in two different files (`ARM_SPEC` in the launcher, `ARM_OF` + the alias
list in the analyzer). Commit `29a194e` exists purely to patch one direction of
this mismatch after the fact.

### S3 — Nondeterministic experiment-directory naming splits a run's telemetry

`train.py:164-168`: `experiment_index = len(glob(f"{args.results_dir}/*"))`. The
directory name is a function of *how many entries already exist*, not of the run.
On a requeue or resume into the same `RESULTS_DIR` (which is stable, since it is
keyed on `SLURM_JOB_ID`), the second attempt creates `001-...` next to `000-...`.
`analyze_image.py:123` globs recursively, finds both, and emits two summary rows
with the *same* `run`/`arm`/`seed` — double-counting one logical run in the
across-seed aggregate. `grad_metrics_file` opening in append mode (`train.py:345`)
partially mitigates the same-dir case while making step sequences non-monotonic
across attempts, which `load_run`'s sort then silently interleaves.

### S3 — Analyzer promises columns that are never emitted

`analyze_image.py:6-8` states Table C reports "throughput, peak memory" and
"nonfinite count". None of the three is present in `record` (`train.py:624-635`)
or computed in `summarize()`. `steps_per_sec` is computed at `train.py:753` but
only goes to the text logger and wandb — not to the structured stream. Peak
memory is never measured. Docstring and implementation have diverged.

### S3 — Reconciler is brittle and reconciles the wrong artifact

`scripts/cluster/reconcile_pipeline.py:42-43` raises `RuntimeError` if `sacct`
returns no row for *any* job id — so one purged job id (sacct retention is finite)
aborts reconciliation of the entire ledger, leaving it stale. It also only ever
touches `pipeline.json`: it never updates `manifest.jsonl`, never records the
results directory, and never checks whether the run's telemetry is complete.

`pipeline.json:active_runs` currently shows `39090540/41/42` as `"pending"`,
submitted 2026-08-13 — i.e. the ledger has not been reconciled since submission.

### S3 — Rank-0-only emission with no world context

Only rank 0 writes. `world_size`, per-rank divergence, and the effective batch
actually consumed are never recorded, so "same config" across arms is asserted by
the launcher rather than observed from the run. Given that label-dropout
contamination (a per-rank conditioning difference) already invalidated nine runs
once, this is the exact class of defect the telemetry should be catching.

---

## 3. Rewiring proposal

The fix is a primary key plus a lifecycle, not more fields. Four changes.

### 3.1 Mint one immutable `run_uid` at submit time and thread it everywhere

`launch_image_phase.py` mints `run_uid` (content hash of the full arm spec +
seed + git_sha + submit timestamp, or simply the existing `run_tag` promoted to
canonical *and made the only name*). It goes into the env block alongside
`RUN_TAG`, the sbatch exports it, `train.py` reads it from the environment and
stamps it on **every** record it writes.

That single change makes the manifest ⋈ metrics ⋈ ledger join a real equality
join instead of a path-parsing heuristic, and deletes the alias table in
`analyze_image.py` entirely. `arm`, `btm_mode`, `fd_k`, `seed`, `git_sha`,
`world_size` come along in the START record — the analyzer never re-derives them
from strings again, so §2's alias bugs become unrepresentable rather than fixed.

Collapse the three vocabularies to one. Keep `ARM_SPEC` in the launcher as the
sole arm registry; the analyzer's display names become a pure presentation map
keyed on the emitted `arm` field.

### 3.2 Make the lifecycle explicit: START / PROGRESS / END

Add `"event"` and `"seq"` to every record. Three event kinds:

- `START` — emitted by `train.py` at rank 0 once the model is built: `run_uid`,
  resolved config, `git_sha`, `world_size`, `SLURM_JOB_ID`, `results_dir`,
  planned `max_steps`. This is the record that ties the run to the manifest.
- `PROGRESS` — the existing per-step records, now discriminated by a `kind`
  field (`grad` vs `probe`) instead of being distinguished by which keys happen
  to be present (§2, S2). Emit them as **one record per step** where possible;
  the probe measurements are of the same step and belong on the same row.
- `END` — emitted from a `finally` block wrapping the training loop (so it fires
  on exception, and via a `SIGTERM`/`SIGUSR1` handler so it fires on preemption
  and timeout): `run_uid`, `last_step`, `planned_steps`, `status`
  (`completed` | `crashed` | `preempted` | `timeout`), exception repr, wall time,
  peak memory. This is the emission that does not currently exist.

The sbatch already knows the exit code (`EXIT_CODE=${PIPESTATUS[0]}`); have it
append a wrapper-level END record too, so a hard kill that beats the signal
handler still produces a terminal event.

### 3.3 Make completeness a precondition of aggregation, not a footnote

`analyze_image.py` gains a gate that runs before any table is printed:

- A run with no `END` record, or `END.status != "completed"`, or
  `last_step < planned_steps`, is **excluded from cross-arm aggregates** and
  listed in an explicit "incomplete runs" section with its step count.
- Windows are defined on the **planned** step axis (fixed absolute step ranges
  shared across arms), not on each run's observed range. "Late" must mean the
  same steps for every arm or it means nothing.
- `arm == "?"` becomes a hard error rather than a printed row.

This converts S1 from a silent wrong answer into a loud refusal — the property
that actually matters for a gated campaign.

### 3.4 Close the loop: one reconciler that updates all three artifacts

Generalize `reconcile_pipeline.py` (or fold it into a BTM-side controller) so a
single `--poll` invocation:

1. Reads `run_uid` → `job_id` from `manifest.jsonl`.
2. Queries `sacct`, tolerating missing rows (mark `unknown`, do not raise).
3. Appends a *new* manifest record per state transition — append-only event log,
   never mutate a prior line — carrying `run_uid`, new status, exit code,
   elapsed, and the resolved `results_dir`.
4. Cross-checks scheduler status against the run's own `END` record and flags
   disagreement (`sacct: COMPLETED` but no `END` in the stream = lost telemetry;
   `END: completed` but `sacct: TIMEOUT` = truncated run misreporting itself).
5. Mirrors the result into `pipeline.json:active_runs`/`completed_runs`.

Step 4 is the check that would have caught the current state, where the ledger
believes three jobs are still `pending` a day after submission.

**Reuse note:** `experiments/direct_energy/campaign.py` already implements most
of this shape — `event()` appending to `events.jsonl` with `at`/`stage`/`status`,
`poll()` diffing scheduler state and emitting transitions, `completed_at` on
terminal states, and a regenerated `summary.md`. The BTM track dropped that
pattern for a write-once manifest and regressed. The controller to build is
`campaign.py` keyed on `run_uid` instead of `stage`, with the in-run START/END
emissions added.

---

## 4. Suggested implementation order

1. **`run_uid` end-to-end** (launcher → env → sbatch → every `train.py` record).
   Unblocks everything else; no behaviour change.
2. **`END` record + signal handler**, and the sbatch-level fallback END.
3. **Analyzer completeness gate + absolute-step windows.** Re-run it over the
   existing `39090540/41/42` telemetry; expect some currently-printed rows to
   move to the "incomplete" section. Any conclusion already drawn from the late-
   window table should be re-derived after this lands.
4. **Split `kind` on the two PROGRESS record types**, then fix
   `update_over_param` (and either emit `param_norm` or delete the column).
5. **Unify the arm vocabulary**; delete the alias table.
6. **Reconciler covering manifest + pipeline + END cross-check.**
7. Deterministic experiment-dir naming (`run_uid` instead of `len(glob(...))`);
   add `world_size`, `steps_per_sec`, `peak_memory`, nonfinite counter to the
   record.

Steps 1–3 are the ones that change whether the campaign's headline table is
trustworthy; 4–7 are hygiene.
