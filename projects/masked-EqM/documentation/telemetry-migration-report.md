# Telemetry migration report — legacy provenance backfill

**Date:** 2026-08-14
**Tools:** `telemetry/legacy.py` (parsers), `telemetry/migrate.py` (backfill),
`telemetry/contradictions.py` (audit)
**Tests:** `tests/test_telemetry_migrate.py` — 54 tests, all passing
**Outputs:** `results/telemetry/` (event logs), `results/telemetry_migration/` (reports)

Reproduce:

```bash
cd projects/masked-EqM
python -m pytest tests/test_telemetry_migrate.py -q
python -m telemetry.contradictions --project-root . --out-dir results/telemetry_migration
python -m telemetry.migrate --project-root . \
    --telemetry-root results/telemetry --report-dir results/telemetry_migration
```

Both tools are **read-only with respect to every legacy artifact**. Verified by
MD5 before and after a full run of both (`.state/pipeline.json`,
`results/btm/manifest.jsonl`, `results_variants.tsv`, both `events.jsonl`) — all
unchanged — and by a test (`test_read_only_with_respect_to_legacy_files`) that
byte-compares the whole tree around a migration.

---

## 1. What was reconstructed

| quantity | value |
| --- | ---: |
| legacy facts parsed | 1446 |
| facts attributable to a job id | 1397 |
| facts **not** attributable (kept, not invented) | 49 |
| distinct job ids | 984 |
| reconstructed logical runs (`run_uid`) | 961 |
| reconstructed executions (`exec_id`) | 993 |
| events emitted | 8657 |

Events by type — note that **no `START` was emitted for any run**:

| event | count | why |
| --- | ---: | --- |
| `OBSERVED` | 1406 | one per ledger assertion. A ledger saying "job X failed" is a third-party observation, not the process reporting its own lifecycle. |
| `NOTICE` | 1273 | 993 reconstruction headers (provenance/confidence/unknowns) + 280 anomaly notices. |
| `PROGRESS` | 5120 | replayed legacy metric streams. |
| `END` | 858 | synthesized, **all with `inferred: true`** and an `inference_basis` listing the ledger locators that justify it. |
| `START` | **0** | nothing in the legacy record establishes that a process started, what config it resolved to, or on how many ranks. Synthesizing one would be fabrication. |

993 executions but only 858 `END`s: **135 executions are deliberately left
unsealed**. Under the lifecycle contract a missing `END` means "this run's
telemetry is not trustworthy", which is the correct and honest state for every
one of them (breakdown in §3).

### Confidence distribution

| confidence | runs | what it means |
| --- | ---: | --- |
| `high` | 12 | BTM manifest rows: real scientific parameters (`btm_mode`, `fd_k`, `fd_eps`, `tc`, `max_steps`, `global_batch`, `seed`), a git sha, a phase, and a corroborating pipeline entry. |
| `medium` | 917 | `pipeline.json` / campaign `jobs` entries: identity, git sha and phase recovered; **scientific parameters unrecoverable**. |
| `low` | 64 | either the arm or the git sha is unknown, or the run's only evidence is a filesystem path. |
| `none` | 0 | — |

### Unknown spec fields (marked, never guessed)

| field | runs affected | why it could not be recovered |
| --- | ---: | --- |
| `params` | 981 | `pipeline.json` records the scientific configuration only as English prose in `description`. There is no machine-recoverable spec. This is the single largest irrecoverable loss. |
| `planned_steps` | 981 | only the BTM manifest records `max_steps`. Without it, **truncation is undetectable** for 981 of 993 executions — a completed run and a run killed at 10% of its schedule are indistinguishable in the legacy record. |
| `seed` | 685 | no declared `seed` field and no strict `_s<N>`/`_seed<N>` suffix on the label. |
| `git_sha` | 22 | absent from the entry, mostly campaign-event-only jobs. |
| `phase` | 17 | absent from the entry. |
| `job_id` | 9 | metric streams whose path carries no `_job<ID>` component. |

Unknown fields are written as the sentinel `"__unknown__"` (seed: `-1`), listed
in `unknown_fields` in `spec.json`, and repeated in every run's reconstruction
header event. No unknown is filled in from a sibling run, a naming convention,
or a plausible default.

### Inferred terminal statuses

| status | executions |
| --- | ---: |
| `completed` | 568 |
| `crashed` (legacy `failed`) | 198 |
| `cancelled` | 86 |
| `timeout` | 6 |
| **no `END` (unsealed)** | **135** |

`failed → crashed` rather than a dedicated state: legacy `failed` covers any
nonzero exit, and `CRASHED` is the honest superset. It is in
`TRUNCATED_STATUSES`, so aggregators refuse to pool it with completed runs —
the right default under uncertainty.

### Anomaly notices attached to the runs

| code | count |
| --- | ---: |
| `schema_drift_missing_keys` | 192 |
| `stranded_non_terminal` | 70 |
| `path_only_identity` | 9 |
| `disputed_terminal_status` | 3 |
| `non_lifecycle_status` | 3 |
| `near_duplicate_keys` | 2 |
| `qualified_terminal_status` | 1 |

### Metric stream replay

9 legacy metric streams were found **in the local repo**, carrying 5120 records:

| assigned `kind` | records |
| --- | ---: |
| `grad` | 4200 |
| `wfb` | 920 |
| `probe` | 0 |
| `unknown` | 0 |

**The `kind` heuristic** (documented in `legacy.classify_metric_record`). The
problem it solves: `gradient_metrics.jsonl` is written by two call sites in
`train.py` that share one file handle and emit no discriminator —

* `train.py:624` writes the full gradient record: `step`, `loss`, `grad_norm`,
  `head_grad_norm`, `backbone_grad_norm`, `max_grad_norm`, `clipped`,
  `learning_rate`, `adaptive_clip`, `weight_decay`;
* `train.py:732` writes the Stage-2 checklist probe: `step`,
  `delta_theta_norm`, `probe_loss_pre`, `probe_loss_post`, `probe_delta_L`,
  `probe_cos_field_update_vs_neg_r`, `field_delta_norm`, `P_t`, `eta_func`.

Both fire on the same `grad_log_every` cadence, so both appear at the same
`step`. The two key sets are **disjoint except for `step`**, so membership in a
witness set is a sound discriminator, not a guess: `_GRAD_WITNESS`, `_PROBE_WITNESS`
and a third `_WFB_WITNESS` for the Stage-3 line-search trace. A record matching
exactly one set gets that `kind`; a record matching **zero or two or more** gets
`kind: "unknown"` with its key signature attached, never a best guess. Every
emitted record carries `kind_basis` naming the witness keys that decided it, so
the classification is auditable without re-running the migration.

One subtlety found while writing the tests: with `--wfb-backward`, `train.py:659`
*adds* `lambda_max`, `lam`, `m_lanczos`, `r_norm` to the ordinary gradient
record. Those keys also occur in the Stage-3 trace, so they cannot witness
either writer — including them would have classified every WFB-enabled training
run as `unknown`, i.e. would have silently discarded exactly the runs the WFB
investigation is about. `_WFB_WITNESS` is restricted to keys exclusive to the
line-search machinery (`eta_star`, `eta_used`, `n_backtracks`, `r_dot_q`, …), and
a test asserts the three witness sets stay pairwise disjoint.

---

## 2. What could NOT be reconstructed, and why

1. **Scientific parameters for 981 of 993 executions.** `pipeline.json` has no
   config field. Its `description` is prose written for a human. Nothing else in
   the repo records what a given historical job actually ran. **This is
   permanent** — it can only be recovered from the cluster-side sbatch
   environment or the run's own output directory, neither of which is in this
   repo.
2. **`planned_steps` for 981 executions**, therefore **truncation is not
   decidable** for them. This is the most damaging single gap: it means no
   historical run can be machine-certified as complete, and any "late training"
   comparison over migrated runs rests on an assumption the data cannot support.
3. **Per-step metrics for essentially the whole history.** Only 9 streams exist
   locally. `gradient_metrics.jsonl` and `fb_direct_metrics.jsonl` — the streams
   the audit brief named — have **zero instances in this repo** (see §5); they
   live on netscratch/holylabs. The migrator supports their format and their
   `<TAG>_job<JOBID>/` path convention (tested against a synthetic fixture) and
   will ingest them when pointed at a results root that contains them.
4. **`last_step`, wall time, peak memory** for every reconstructed run — no
   ledger records them. Written as `last_step: -1` with the field listed in the
   `END` record's own `unknown_fields`.
5. **Terminal outcome for 135 executions.** Broken down in §3.
6. **49 facts could not be attributed to any job.** Campaign events that name no
   job id, and the 7 non-job entries polluting the longer-training `jobs` dict.
   They are preserved verbatim in
   `results/telemetry_migration/unattributed_facts.jsonl` rather than being
   attached to a plausible neighbour.
7. **The identity of 9 metric streams.** `results/direct_energy_campaign/stage3_pilot/*/metrics.jsonl`
   and `results/wfb_stage3/*_metrics.jsonl` have no job id anywhere in their
   path. They migrate as their own `confidence: low` runs with `job_id` marked
   unknown and a `path_only_identity` notice. They are **not** attached to a job
   by name resemblance — path-based attribution is the non-injective decoding the
   whole identity scheme exists to abolish.

---

## 3. Why 135 executions were left without an `END`

| reason | executions |
| --- | ---: |
| stranded at a non-terminal status inside `completed_runs` (`pending` 68, `running` 2) | 70 |
| no lifecycle status recorded by any source (manifest `submitted`-only rows, campaign-event-only jobs, the 4 live `active_runs`, path-only metric streams) | 59 |
| terminal status **disputed** between sources | 3 |
| status is an annotation, not a lifecycle state (`superseded`, `INVALID -- discarded`) | 3 |
| **total** | **135** |

The disputed three are jobs **35436507, 35436518, 35436519**, each appearing
twice in `completed_runs` — once `failed`, once `completed`. The migrator emits
both claims as `OBSERVED` events, adds an error-level `NOTICE` with code
`disputed_terminal_status` listing both locators, and **synthesizes no `END`**.
Picking a winner would manufacture certainty the data does not contain; leaving
the run unsealed makes it read as untrustworthy, which is exactly what it is.

Conversely, one qualified case *is* sealed: the single
`completed_with_write_error` entry seals as `completed` with the caveat carried
in the `END` record's `caveats` field and a `qualified_terminal_status` notice —
the process did finish; the caveat bears on its *outputs*, a different question.

---

## 4. Definitive contradiction table

`results/telemetry_migration/contradictions.{json,md}`. **1109 findings** over
1446 facts and 984 distinct job ids; 181 job ids appear in more than one
artifact.

### By severity

| severity | count |
| --- | ---: |
| critical | 15 |
| major | 175 |
| minor | 919 |

### All 15 critical findings

| # | job id | finding |
| ---: | --- | --- |
| 1 | — | **70 entries sit in `completed_runs` at a non-terminal status** (`pending` 68, `running` 2). The ledger never learned their outcome. |
| 2 | — | **All 12 BTM manifest rows are frozen at `status: "submitted"`.** A mutable cell written once at submission and never updated; it carries no information about any run's outcome. (Cross-referencing `pipeline.json` shows all 12 were in fact `cancelled` — which is how the migration seals them.) |
| 3 | `35335934` | `job_id` addresses 2 distinct `completed_runs` entries. |
| 4 | `35436507` | `job_id` addresses 2 distinct entries. |
| 5 | `35436518` | `job_id` addresses 2 distinct entries. |
| 6 | `35436519` | `job_id` addresses 2 distinct entries. |
| 7 | `35436507` | **conflicting terminal statuses: `completed` vs `crashed`(legacy `failed`).** |
| 8 | `35436518` | conflicting terminal statuses: `completed` vs `crashed`. |
| 9 | `35436519` | conflicting terminal statuses: `completed` vs `crashed`. |
| 10 | `35335932` | git sha disagreement: `696c49c` vs `f5c024c`. |
| 11 | `35335933` | git sha disagreement: `696c49c` vs `f5c024c`. |
| 12 | `35335934` | git sha disagreement: `696c49c` vs `f5c024c`. |
| 13 | `36142162` | git sha disagreement: `7f1521d` vs `c100fcf`. |
| 14 | `36142163` | git sha disagreement: `7f1521d` vs `c100fcf`. |
| 15 | `36187413` | git sha disagreement: `7b6e5a3` vs `92779cf`. |

A git-sha disagreement is critical because it means two ledgers believe the same
job ran different code: any claim about what that job tested depends on which
file the reader happened to open. (Short-vs-long forms of the same sha are
collapsed and not reported.)

### Major findings (175)

| kind | count | detail |
| --- | ---: | --- |
| `label` disagreement | 134 | the same job id is named differently by different artifacts (`pipeline.run_id` vs campaign stage name vs manifest `run_tag`). This is why `job_id` is the only usable join key. |
| duplicate `run_id` | 34 | 34 distinct `run_id` values each address 2–5 `completed_runs` entries. A lookup by `run_id` returns an arbitrary one of them. |
| `status_enum` violations | 3 | `"INVALID -- discarded"` (2 entries), `completed_with_write_error` (1), `superseded` (1). |
| `jobs` dict inverted | 1 | `direct_energy_longer_training/status.json` maps `name -> job_id` while `direct_energy_campaign` maps `job_id -> object`. A reader written for one shape reads 113 phantom jobs named after arms from the other. |
| `jobs` dict polluted | 1 | 7 entries in that dict are not jobs (`epoch08_fid_none_value: 129.12085939165866`, `epoch15_checkpoints: "complete"`, …). |
| event-log membership drift | 1 | `direct_energy_campaign/events.jsonl` has 65 records, `status.json['events']` has 60; **5 only in the file, 0 only in status.json**. Two views of one campaign that are not the same view. |
| event ordering | 1 | 1 backwards timestamp transition in that file (`21:10:00` → `21:00:00`). An append-only log whose clock goes backwards cannot be ordered by time. |

### Minor findings (919)

* **803 orphans** — job ids known to exactly one artifact: 794 `pipeline.completed_runs`-only,
  4 `pipeline.active_runs`-only, 4 `direct_energy_longer_training.events`-only,
  1 `direct_energy_campaign.events`-only. **Zero manifest-only orphans on the
  historical 12** (see §5, item 3).
* **105 `phase` disagreements** between a pipeline `phase` and a campaign `stage`.
* **schema drift in `pipeline.json`**: 191/927 entries lack `expected_runtime`,
  287/927 lack `exit_code`, 525/927 lack `final_metric`.
* **5 near-duplicate key pairs**: `ckpt`(18)/`checkpoint`(6),
  `note`(17)/`analysis_note`(2), `superseded_job_id`(11)/`supersedes`(1),
  `started_at`(1)/`submitted_at`(911), `checkpoint_direct`(1)/`checkpoint_none`(1).
  Reported, never merged — no evidence establishes they are synonyms.
* **event-log entropy**: `direct_energy_longer_training/events.jsonl` has **75
  distinct key-shapes over 119 records**, plus two competing timestamp keys
  (`at` 45 / `timestamp` 74) and two job keys (`job` 51 / `jobs` 58). It has no
  schema.

### Artifact coverage

| artifact | facts | distinct job ids |
| --- | ---: | ---: |
| `pipeline.completed_runs` | 923 | 919 |
| `pipeline.active_runs` | 4 | 4 |
| `btm.manifest` | 12 | 12 |
| `direct_energy_campaign.status.jobs` | 26 | 26 |
| `direct_energy_campaign.events` | 66 | 29 |
| `direct_energy_campaign.status.events` | 60 | 26 |
| `direct_energy_longer_training.status.jobs` | 113 | 106 |
| `direct_energy_longer_training.events` | 221 | 124 |
| `results_variants.tsv` | 21 | 20 |

---

## 5. Corrections to the audit brief

Everything in the brief was verified against the files. Five items were wrong or
imprecise; the rest were confirmed exactly.

1. **"70 duplicates, worst offenders repeated 5x" — imprecise.** 923 entries over
   853 distinct `run_id` is **70 excess entries**, but they are spread over
   **34 distinct duplicated `run_id` values** (2–5 entries each; five values
   appear 5× — `corruption_sanity_downsample_only_seed2`,
   `corruption_sanity_downsample_1to1_seed0`, `corruption_sanity_downsample_1to4_seed1`,
   `corruption_sanity_downsample_1to4_seed2`, `corruption_sanity_fourier_1to4_seed2`).
   "70 duplicates" reads as 70 duplicated keys; it is 34.

2. **Schema-drift counts were computed over `completed_runs` only.** Over
   `completed_runs` (923) the brief's numbers are exact: `expected_runtime`
   missing 191, `final_metric` 521, `exit_code` 283. Over **all 927** entries
   (including `active_runs`) they are 191 / **525** / **287**. The report above
   uses the 927 basis.

3. **"manifest `run_tag` and pipeline `run_id` differ, so only `job_id` joins
   them" — wrong for the actual data, and the truth is worse.** For all 12 jobs
   the manifest covers, the two strings are **identical**
   (`btm_IIA_btm_scalar_exact_s0` in both). The `G`/`D1`/`D4`/`V` alias
   vocabulary belongs to a **disjoint set of jobs** — the four clean reruns
   `39090540/41/42/39090739` in `active_runs`. Those four have **no manifest row
   at all**. So the two vocabularies never meet on a shared job id, and the
   campaign's *current* runs are unrecorded by the campaign's own manifest.
   (This also means the migration produced **zero** manifest-vs-pipeline label
   conflicts, not twelve.)

4. **"`gradient_metrics.jsonl` and `fb_direct_metrics.jsonl` under results dirs"
   — neither file exists anywhere in this repo.** `find` returns 0 of each. The
   9 metric streams that do exist locally are
   `results/direct_energy_campaign/stage3_pilot*/*/metrics.jsonl`,
   `results/direct_energy_campaign/stage2_fixed_batch/metrics.jsonl` and
   `results/wfb_stage3/*_metrics.jsonl`. The named files live on cluster storage.
   Consequence: the two-shapes-one-file defect could not be exercised against
   real data — the replay produced 4200 `grad` and 0 `probe` records. The
   heuristic is implemented, documented, and unit-tested against the exact record
   shapes emitted by `train.py:624` and `train.py:732`, but it has **not** been
   validated on a real interleaved file.

5. **"one is out of time order" in `direct_energy_campaign/events.jsonl` — confirmed,
   and it is one of the 5 hand-inserted lines.** Line index 4
   (`2026-07-23T21:00:00+00:00`, stage `0_evaluator_gpu_regression`) follows line 3
   (`21:10:00`). All five extra lines are hand-written in a different JSON style
   (no spaces after separators) and none appear in `status.json['events']`; the
   drift is one-directional (0 records exist only in `status.json`).

Confirmed exactly as stated: 4 `active_runs` / 923 `completed_runs`; 853 distinct
`run_id`; the four duplicated job ids `35436507`/`35436518`/`35436519`/`35335934`
with the first three carrying conflicting statuses and the fourth not; the status
enum violations and 70 stranded non-terminal entries; 16 ad-hoc keys; the
manifest's 12 rows all frozen at `submitted` and all in fact cancelled; the
campaign's 65-vs-60 event split; the inverted and float-polluted longer-training
`jobs` dict (7 non-job values); 119 events across 75 shapes (the brief said 76 —
75 by exact key-tuple, an off-by-one in counting method, not a substantive
difference) with `at`/`timestamp` and `job`/`jobs` both in use; the 21-row
`results_variants.tsv` with no BTM campaign rows.

**On "64 known contradictions":** no natural grouping of the findings reproduces
64. The checker reports 1109 findings, of which 15 critical and 175 major. The
number 64 is not derivable from any subset the tool computes; if it referred to a
specific hand-built list, that list is not in the repo.

---

## 6. Design decisions worth knowing before extending this

* **Ledger facts are `OBSERVED`, never `START`.** Emitting `START` for a run that
  never reported starting would make a reconstruction structurally
  indistinguishable from a natively instrumented run. Every reconstructed stream
  begins instead with a `NOTICE` carrying `provenance`, `confidence`,
  `unknown_fields` and `primary_source`.
* **Arms are carried verbatim, never decoded.** Only a strict `_s<N>`/`_seed<N>`
  suffix is stripped. No mapping onto a canonical arm vocabulary — that mapping is
  precisely the non-injective decoding that once averaged
  `btm_scalar_fd_directional` and `btm_scalar_fd_directional4` together. Cost: the
  same experiment under two ledger names gets two `run_uid`s. That is an
  over-count, it is visible in the contradiction report, and it is the safe
  direction to err.
* **Campaign-level facts are not broadcast onto jobs.** An early version attached
  `direct_energy_longer_training/status.json`'s top-level `commit` to each of its
  106 jobs, which manufactured **103 false git-sha "disagreements"** and buried
  the 6 real ones. The campaign-level `commit` and `started_at` now live in the
  observation's `raw` payload as context only. Generally: a fact recorded at
  campaign scope is not evidence at job scope.
* **Job-id regexes are anchored.** `"epoch08_fid_none_value": 129.12085939165866`
  sits inside a dict named `jobs`; an unanchored `\d{6,}` mines the phantom job
  `12085939` out of its mantissa. Tested.
* **Idempotence is by content digest of the written bytes**, not by file
  existence. An interrupted migration leaves a truncated stream whose recorded
  digest still "matches"; an existence check would let that truncation stand
  forever. Second run reports 993/993 `unchanged` and writes nothing.
* **Stale output is pruned.** When the reconstruction logic changes, a run moves
  to a new `run_uid` and its old directory would otherwise persist — leaving two
  reconstructions of one job, silently disagreeing, which is the disease being
  cured. Pruning is guarded by a `.migrator-owned` marker, so pointing the
  migrator at a directory it did not create is a no-op rather than a deletion.

---

## 7. Recommended follow-up

1. **Point the migrator at cluster storage** (`--project-root` with a results
   root containing `<TAG>_job<JOBID>/` dirs) to ingest the real
   `gradient_metrics.jsonl` files and validate the `grad`/`probe` split on
   genuinely interleaved data. Until then that split is tested but not
   field-validated.
2. **Do not aggregate migrated runs by `planned_steps`.** 981 of 993 have none,
   so truncation is undecidable for them.
3. **`results_variants.tsv` should be regenerated as a fold over the event log**
   rather than hand-maintained. It currently has no code reader and no code
   writer and omits the entire BTM campaign.
4. **The BTM manifest should record the 4 clean reruns** (`39090540/41/42`,
   `39090739`) or be retired in favour of the event log — as of now the campaign's
   only live runs are absent from the campaign's own manifest (§5, item 3).
5. **Wire `contradictions.py` into CI** against a stored baseline: a rising count
   is a regression in provenance hygiene, and the JSON output diffs cleanly.
