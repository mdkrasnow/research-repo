# Research Repo — Project Structure Contract (Dispatch/Pipeline Integration)

This README defines the **required project structure** for any `projects/<slug>/` directory to be compatible with the `/dispatch` pipeline, run orchestration, and SLURM/remote execution flow.

If a project matches this contract, `/dispatch` can:
- plan and implement tasks
- run experiments (local and/or SLURM)
- track runs + results
- update state deterministically across iterations

---

## 1) Project Location + Naming

### Required
- **Path**: `projects/<slug>/`
- **Slug**: lowercase, kebab-case recommended (`my-project`, `ablation-v2`)

### Recommended
- One project = one research question (or tightly coupled set of experiments).

---

## 2) Required Directory Tree

Every project must include the following minimum structure:

```

projects/<slug>/
.state/
pipeline.json
lock/                       (created at runtime; do not commit contents)
documentation/
implementation-todo.md
debugging.md
queue.md
experiments/                  (Python scripts, or other runnable code)
configs/                      (experiment configs, e.g., json/yaml)
slurm/
jobs/                       (sbatch scripts)
logs/                       (runtime logs; can be empty initially)
runs/                         (created at runtime; run ledgers + outputs)
results/
summary.md                  (project-level results summary; can be stub)

````

### Notes
- `runs/` may start empty but must exist (or `/dispatch` will create it).
- `slurm/logs/` can be empty but must exist.
- `.state/lock/` is created by locking scripts and should not be manually edited.

---

## 3) Required Files + Types + Contents

### A) `.state/pipeline.json` (JSON; REQUIRED)
**Purpose:** the authoritative state machine for the project.

**Type:** JSON object

**Required keys and types:**
```json
{
  "project": "string",
  "phase": "string",
  "next_action": {
    "command": "string",
    "reason": "string",
    "hint": "string"
  },
  "needs_user_input": {
    "value": "boolean",
    "prompt": "string"
  },
  "active_runs": [
    {
      "run_id": "string",
      "experiment": "string",
      "status": "string",
      "job_id": "string|null"
    }
  ],
  "slurm_wait": {
    "next_poll_after": "string|null",
    "poll_interval_seconds": "number"
  },
  "events": [
    {
      "id": "string",
      "ts": "string (ISO8601 UTC recommended)",
      "action": "string",
      "detail": "string (optional)"
    }
  ]
}
````

**Phase values (recommended enum):**

* `INIT`
* `IMPLEMENT`
* `RUN`
* `WAIT_SLURM`
* `CHECK_RESULTS`
* `DEBUG`
* `DONE`

**Rules:**

* `phase` should always reflect the *current* stage.
* `next_action.command` should be a valid slash command (e.g., `/run-experiments`, `/check-results`, `/dispatch`).
* `needs_user_input.value=true` means the pipeline must pause until the user satisfies the prompt.
* `active_runs` lists runs that are pending or running.
* `slurm_wait.next_poll_after` controls polling cadence. If set in the future, `/dispatch` should skip until due.
* `events` is append-only for traceability.

---

### B) `documentation/implementation-todo.md` (Markdown; REQUIRED)

**Purpose:** source of truth for implementation tasks. `/dispatch` uses this to decide what to implement next and to mark completion.

**Type:** Markdown

**Required sections:**

* A list of tasks (checkboxes)
* A place to record completion notes

**Minimal recommended format:**

```markdown
# Implementation TODO — <Project Name>

## Ready
- [ ] (T1) <task>
- [ ] (T2) <task>

## Completed
- [x] (T0) <task> (optional notes)
```

**Rules:**

* Each task must have a stable identifier like `(T1)`.
* `/dispatch` may move tasks between sections but should preserve task text.
* If tasks are blocked, add a `## Blocked` section and explain why.

---

### C) `documentation/debugging.md` (Markdown; REQUIRED)

**Purpose:** structured bug/issue tracker for the pipeline. `/dispatch` can read it to decide when to shift to DEBUG, and write updates when failures occur.

**Type:** Markdown

**Required content:**

* A list of current issues and their status
* A place to record root cause and next step

**Minimal recommended format:**

```markdown
# Debugging — <Project Name>

## Active Issues
- [ ] (D1) <issue summary>
  - Symptoms:
  - Suspected cause:
  - Next step:
  - Logs/paths:

## Resolved
- [x] (D0) <issue summary> — <resolution note>
```

**Rules:**

* Stable identifiers like `(D1)` are required.
* Any failure (tests, experiment run, missing dependency, slurm errors) should create or update an active issue here.
* If debugging blocks progress, `pipeline.needs_user_input.value` may be set true with the next step.

---

### D) `documentation/queue.md` (Markdown; REQUIRED)

**Purpose:** defines the ordered list of experiments to run. `/run-experiments` and `/dispatch` read this to know what to submit next.

**Type:** Markdown

**Required content per experiment item:**

* unique ID (e.g., `Q-001`)
* hypothesis / goal
* config file path
* resource requirements (CPU/GPU, time)
* status (READY / IN_PROGRESS / DONE / FAILED)

**Minimal recommended format:**

```markdown
# Experiment Queue — <Project Name>

## READY
### Q-001: <title>
- Hypothesis: <string>
- Config: `configs/q001.json`
- Resources: CPU only, <estimate>
- Notes: <optional>

## IN_PROGRESS
## DONE
## FAILED
```

**Rules:**

* Each item **must** reference a config path under `configs/`.
* If submitted to SLURM, `/dispatch` should add run directory reference and mark IN_PROGRESS.
* When results are finalized, move to DONE and add a one-line summary.

---

### E) `results/summary.md` (Markdown; REQUIRED)

**Purpose:** human-facing summary of results across experiments. Updated by `/check-results`.

**Type:** Markdown

**Minimum stub:**

```markdown
# Results Summary — <Project Name>

## Key Findings
- TBD

## Completed Experiments
- None yet
```

---

## 4) Required Code/Execution Assets

### A) `experiments/` (Directory; REQUIRED)

**Purpose:** runnable experiment entrypoints.

**Allowed file types:**

* `.py` (recommended)
* any runnable scripts you support (but must be callable from sbatch scripts)

**Contract:**

* There must exist **at least one** runnable file referenced by queue/configs.
* For Python: strongly recommend a CLI pattern:

  * `python experiments/<name>.py --config configs/<file>.json`

---

### B) `configs/` (Directory; REQUIRED)

**Purpose:** structured configs for experiments.

**Allowed file types:**

* `.json` (recommended)
* `.yaml` / `.yml` (allowed)

**Contract:**

* Every queue item must reference a config file here.
* Config should include enough information to reproduce results deterministically (seed, dataset, parameters).

---

### C) `slurm/jobs/` (Directory; REQUIRED)

**Purpose:** SBATCH scripts used to submit queue items to SLURM.

**Allowed file types:**

* `.sbatch` (required for SLURM submission)

**Contract:**

* Each queued experiment that will run on SLURM must have a corresponding `.sbatch` script.
* Script must write logs to `slurm/logs/` and write outputs into the associated run directory (recommended).

**Minimum expectations inside `.sbatch`:**

* SLURM headers (`#SBATCH ...`)
* `cd` into repo root or project directory
* call the experiment with config

---

### D) `slurm/logs/` (Directory; REQUIRED)

**Purpose:** centralized log location pulled back from cluster (if remote mode).

**File types:**

* `.out`, `.err`, `.log`, text logs

**Contract:**

* `/check-results` expects to find logs here for parsing.

---

## 5) Runtime-Generated Artifacts (created by pipeline)

### A) `runs/<run_id>/submit.json` (JSON; generated)

**Purpose:** submission ledger for a single run.

**Type:** JSON

**Expected keys:**

```json
{
  "run_id": "string",
  "experiment": "string",
  "config": "string",
  "sbatch_script": "string",
  "submitted_at": "string",
  "submitted_by": "string",
  "status": "string",
  "job_id": "string|null",
  "notes": "string"
}
```

### B) `runs/<run_id>/results.md` (Markdown; generated)

**Purpose:** summarized results for that run (key outputs, metrics, interpretation).

### C) `runs/<run_id>/artifacts/` (Directory; optional)

**Purpose:** plots, tables, checkpoints.

---

## 6) Pipeline Behavior Assumptions (Important)

### What `/dispatch` expects to be true

* It can read + write:

  * `.state/pipeline.json`
  * `documentation/*.md`
  * create `runs/` subfolders
* It can locate queue items in `documentation/queue.md`
* It can locate config paths under `configs/`
* It can find sbatch scripts under `slurm/jobs/` if SLURM execution is required

### What `/run-experiments` expects to be true

* There is at least one `READY` queue item.
* That queue item has a config file and (if SLURM) a `.sbatch`.
* The sbatch script is consistent with the config path and writes logs.

### What `/check-results` expects to be true

* `pipeline.active_runs` contains job IDs (if SLURM).
* Logs exist (local or synced) under `slurm/logs/`.
* It can write results into `runs/<run_id>/results.md` and update `results/summary.md`.

---

## 7) Recommended Optional Files

These are not required for compatibility but strongly recommended:

* `README.md` (project overview: question, method, how to run)
* `tests/` (pytest etc. for core invariants)
* `data/` (if local data required; avoid committing large data)
* `notebooks/` (analysis exploration)

---

## 8) Minimal Example Checklist (Copy/Paste)

To make a new project compatible:

* [ ] Create `projects/<slug>/`
* [ ] Add required folders:

  * [ ] `.state/`
  * [ ] `documentation/`
  * [ ] `experiments/`
  * [ ] `configs/`
  * [ ] `slurm/jobs/`
  * [ ] `slurm/logs/`
  * [ ] `runs/`
  * [ ] `results/`
* [ ] Create required files:

  * [ ] `.state/pipeline.json`
  * [ ] `documentation/implementation-todo.md`
  * [ ] `documentation/debugging.md`
  * [ ] `documentation/queue.md`
  * [ ] `results/summary.md`
* [ ] Add at least one experiment script in `experiments/`
* [ ] Add at least one config in `configs/`
* [ ] Add sbatch in `slurm/jobs/` if queue item requires SLURM
* [ ] Set pipeline `phase` and `next_action` appropriately (usually `IMPLEMENT` or `RUN`)

---

## 9) Default Starter `pipeline.json` Template

Use this as a safe starting point:

```json
{
  "project": "<slug>",
  "phase": "INIT",
  "next_action": {
    "command": "/dispatch",
    "reason": "Initialize project pipeline",
    "hint": "Create initial tasks and queue"
  },
  "needs_user_input": { "value": false, "prompt": "" },
  "active_runs": [],
  "slurm_wait": { "next_poll_after": null, "poll_interval_seconds": 900 },
  "events": [{ "id": "evt-0001", "ts": "YYYY-MM-DDTHH:MM:SSZ", "action": "INIT" }]
}
```

---

## 10) What breaks integration (common pitfalls)

* Missing any required file (especially `pipeline.json`, queue, debugging, todo)
* Queue items without config references
* SLURM jobs not writing logs anywhere predictable
* No stable IDs (T1, D1, Q-001)
* Phase and `next_action` out of sync (e.g., phase RUN but queue empty)

---

## Summary

If your project includes the required directories + files with the expected types and minimal structure, the dispatch pipeline can operate autonomously and deterministically—implementing tasks, submitting experiments (local or remote SLURM), polling results, updating documentation, and maintaining durable state without projects stepping on each other.