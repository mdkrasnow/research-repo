---
description: Run autonomous research loop - hypothesize, code, run, measure, keep/revert, repeat. Karpathy-style autoresearch.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, Task, Agent
argument-hint: --project <slug> [--iterations N] [--dry-run] [--local]
---
# /autoresearch

Run an autonomous, continuous research loop on a project. The agent forms hypotheses, modifies code, runs experiments, measures results, and keeps or reverts changes — all without human intervention.

## Usage

```bash
/autoresearch --project my-experiment              # Run autoresearch loop
/autoresearch --project my-experiment --iterations 20  # Limit to 20 iterations
/autoresearch --project my-experiment --dry-run     # Preview first iteration only
/autoresearch --project my-experiment --local       # Force local execution (no SLURM)
```

## Prerequisites

The project MUST have a `program.md` governance file at `projects/<slug>/program.md`. This file defines:
- **Objective metric** and how to measure it
- **Constraints** (files allowed, time budgets)
- **Ratchet rules** (when to keep vs revert)
- **Termination conditions**

If `program.md` doesn't exist, prompt the user to create one (offer to generate from template at `templates/program.md`).

## The Autoresearch Loop

```
┌─────────────────────────────────────────────────┐
│  AUTORESEARCH LOOP (runs until termination)     │
│                                                 │
│  1. READ CONTEXT                                │
│     ├─ program.md (governance)                  │
│     ├─ results.tsv (experiment history)         │
│     └─ Current code state                       │
│                                                 │
│  2. HYPOTHESIZE                                 │
│     ├─ Analyze past results                     │
│     ├─ Identify promising direction             │
│     └─ Form specific, testable hypothesis       │
│                                                 │
│  3. IMPLEMENT                                   │
│     ├─ Modify ONLY files listed in              │
│     │   program.md:files_allowed                │
│     ├─ Make ONE change (isolate variables)      │
│     └─ Git commit with hypothesis description   │
│                                                 │
│  4. RUN                                         │
│     ├─ Local: eval_command with timeout          │
│     └─ SLURM: submit job, poll until complete   │
│                                                 │
│  5. MEASURE                                     │
│     ├─ Parse metric from output/logs            │
│     └─ Compare to best_so_far                   │
│                                                 │
│  6. RATCHET DECISION                            │
│     ├─ IMPROVED: Keep commit, update best       │
│     ├─ REGRESSED: git revert, log failure       │
│     └─ CRASHED: git revert, log crash           │
│                                                 │
│  7. LOG                                         │
│     ├─ Append to results.tsv                    │
│     └─ Update pipeline.json                     │
│                                                 │
│  8. CHECK TERMINATION                           │
│     ├─ Max iterations reached?                  │
│     ├─ Target metric achieved?                  │
│     ├─ Plateau detected?                        │
│     ├─ Max wall hours exceeded?                 │
│     └─ Max consecutive failures?                │
│                                                 │
│  → If not terminated: go to step 1              │
└─────────────────────────────────────────────────┘
```

## Detailed Step Instructions

### Step 1: Read Context

```
1. Read projects/<slug>/program.md
   - Parse all governance fields
   - Validate required fields exist (metric, direction, eval_command or eval_grep)

2. Read projects/<slug>/results.tsv (create if missing)
   - Header: iteration | metric | status | description | git_sha | timestamp
   - Parse all past results to understand what's been tried

3. Read current state of allowed files
   - Understand what the code currently does
   - Identify what hasn't been tried yet

4. Read projects/<slug>/.state/pipeline.json
   - Check phase (should be AUTORESEARCH or will be set to it)
   - Check for any blocking issues
```

### Step 2: Hypothesize

**CRITICAL**: The agent must form its OWN hypothesis. Do NOT ask the user what to try.

```
Based on results.tsv and program.md:exploration_dimensions:

1. If results.tsv is empty (first run):
   - Run the baseline configuration as-is
   - This establishes the starting metric

2. If results.tsv has entries:
   - Analyze which dimensions have been explored
   - Identify which changes improved vs regressed
   - Form a NEW hypothesis that hasn't been tried
   - Write a 1-line description of what you're testing

3. Hypothesis quality rules:
   - ONE change per experiment (isolable)
   - Motivated by past results (not random)
   - Specific and testable ("increase lr from 1e-3 to 3e-3")
   - Not a repeat of a previously failed approach
```

### Step 3: Implement

```
1. Modify ONLY files in program.md:files_allowed
   - NEVER modify files in program.md:files_readonly
   - NEVER modify program.md itself

2. Make the minimum change to test the hypothesis
   - Prefer config changes over code changes
   - Prefer small code changes over large ones
   - If two approaches are equivalent, choose the simpler one

3. Git commit with descriptive message:
   - Format: "autoresearch(<slug>): <hypothesis description>"
   - Example: "autoresearch(ired): increase lr from 1e-3 to 3e-3"
```

### Step 4: Run Experiment

**Local mode** (`mode: local` in program.md):
```bash
# Run with timeout
timeout <max_runtime_seconds> <eval_command>
```

**SLURM mode** (`mode: slurm` in program.md):
```
1. Ensure SSH session: scripts/cluster/ensure_session.sh
2. Submit job: scripts/cluster/remote_submit.sh
   - Use pilot_steps for rapid iteration (not full_steps)
   - Use partition from program.md
   - Capture job_id
3. Poll until completion:
   - First poll at 60 seconds (catch init errors)
   - Then every 2 minutes
   - Timeout at max_slurm_minutes
4. Fetch logs: scripts/cluster/remote_fetch.sh
```

**IMPORTANT for SLURM mode**: Use `pilot_steps` (short runs) for the autoresearch loop. Only scale to `full_steps` for validated improvements (manually, after autoresearch completes).

### Step 5: Measure

```
1. Extract metric from output:
   - Local: parse stdout for eval_grep pattern
   - SLURM: parse log file for eval_grep pattern

2. Handle edge cases:
   - Metric not found in output → status: CRASH
   - Multiple metric values → use the LAST one (final evaluation)
   - NaN or Inf → status: CRASH
   - Job failed (non-zero exit) → status: CRASH

3. Compare to program.md:best_so_far
   - If best_so_far is null: this is the baseline, always KEEP
   - If direction=minimize: improvement = (best_so_far - metric) > keep_threshold
   - If direction=maximize: improvement = (metric - best_so_far) > keep_threshold
```

### Step 6: Ratchet Decision

```
KEEP (metric improved):
  1. Update program.md:best_so_far = metric
  2. Update program.md:best_commit = current git SHA
  3. Log to results.tsv: status=KEEP
  4. Reset consecutive_failures counter to 0
  5. Print: "✓ KEEP: <metric> (improved from <old>)"

REVERT (metric regressed):
  1. Run: git revert HEAD --no-edit
  2. Log to results.tsv: status=REVERT
  3. Increment consecutive_failures counter
  4. Print: "✗ REVERT: <metric> (worse than <best>)"

CRASH (experiment failed):
  1. Run: git revert HEAD --no-edit
  2. Log to results.tsv: status=CRASH
  3. Increment consecutive_failures counter
  4. Print: "⚠ CRASH: <error_summary>"
  5. If error is fixable (import error, typo): fix and retry ONCE
```

**IMPORTANT**: When reverting, use `git revert HEAD --no-edit` (creates a new revert commit) rather than `git reset HEAD~1` to preserve full history. The revert commit message should include why: "Revert autoresearch(<slug>): <hypothesis> — metric regressed from X to Y".

### Step 7: Log Results

```
1. Append to projects/<slug>/results.tsv:
   <iteration> | <metric_value> | <KEEP|REVERT|CRASH> | <hypothesis_description> | <git_sha> | <ISO_timestamp>

2. Update projects/<slug>/.state/pipeline.json:
   - phase: "AUTORESEARCH"
   - autoresearch.iteration: <current>
   - autoresearch.best_metric: <best_so_far>
   - autoresearch.best_commit: <sha>
   - autoresearch.consecutive_failures: <count>
   - autoresearch.total_keeps: <count>
   - autoresearch.total_reverts: <count>
   - autoresearch.started_at: <ISO timestamp>
   - autoresearch.last_iteration_at: <ISO timestamp>

3. Print iteration summary:
   "Iteration <N>/<max>: <metric> (<status>) | Best: <best> | Keeps: <k> Reverts: <r>"
```

### Step 8: Check Termination

```
Check each condition in order:

1. max_iterations reached?
   → "Terminating: max iterations (<N>) reached. Best: <metric> at <commit>"

2. target_metric achieved?
   → "Terminating: target metric (<target>) achieved! Final: <metric>"

3. max_consecutive_failures reached?
   → "Terminating: <N> consecutive failures. Best: <metric>. Consider revising program.md."

4. plateau detected? (stop_on_plateau && no improvement in plateau_window iterations)
   → "Terminating: plateau detected (<N> iterations without improvement). Best: <metric>"

5. max_wall_hours exceeded?
   → "Terminating: wall-clock limit (<H>h) exceeded. Best: <metric>"

If none triggered: CONTINUE to next iteration.

On termination:
  1. Update pipeline.json: phase → "CHECK" (human reviews results)
  2. Update pipeline.json: autoresearch.terminated_reason = <reason>
  3. Print final summary:
     "AUTORESEARCH COMPLETE
      Iterations: <N>
      Best metric: <value> (started at <baseline>)
      Improvement: <percentage>%
      Keeps: <k> / Reverts: <r> / Crashes: <c>
      Best commit: <sha>
      Duration: <hours>h <minutes>m"
  4. Commit results.tsv: "autoresearch(<slug>): complete — <N> iterations, <metric> best"
```

## Pipeline.json Schema Extension

When autoresearch is active, pipeline.json gains an `autoresearch` field:

```json
{
  "phase": "AUTORESEARCH",
  "autoresearch": {
    "active": true,
    "iteration": 15,
    "best_metric": 0.00823,
    "best_commit": "abc1234",
    "baseline_metric": 0.00977,
    "consecutive_failures": 2,
    "total_keeps": 8,
    "total_reverts": 5,
    "total_crashes": 2,
    "started_at": "2026-03-31T10:00:00Z",
    "last_iteration_at": "2026-03-31T14:30:00Z",
    "terminated_reason": null
  }
}
```

## Results.tsv Format

Tab-separated file at `projects/<slug>/results.tsv`:

```
iteration	metric	status	description	git_sha	timestamp
1	0.00977	KEEP	baseline run	abc1234	2026-03-31T10:05:00Z
2	0.01200	REVERT	increase lr from 1e-3 to 5e-3	def5678	2026-03-31T10:15:00Z
3	0.00950	KEEP	add weight decay 0.01	ghi9012	2026-03-31T10:25:00Z
4	0.00823	KEEP	reduce hidden_dim from 256 to 128	jkl3456	2026-03-31T10:35:00Z
```

## Integration with Existing System

- **Autoresearch IS a dispatch mode**: When `phase=AUTORESEARCH`, `/dispatch` delegates to the autoresearch loop
- **Ralph loop compatible**: Ralph's stop hook recognizes AUTORESEARCH phase as actionable
- **Lock-aware**: Each iteration acquires/releases the project lock
- **Git-disciplined**: Every change is committed; reverts are clean revert commits
- **SLURM-aware**: Uses existing `scripts/cluster/*` infrastructure
- **Results persist**: results.tsv and pipeline.json survive across sessions

## Resuming Autoresearch

If a session ends mid-autoresearch (crash, user interrupt, etc.):

1. `/autoresearch --project <slug>` reads pipeline.json
2. Sees `phase=AUTORESEARCH` with `autoresearch.active=true`
3. Reads `autoresearch.iteration` to know where it left off
4. Reads `results.tsv` for full experiment history
5. Continues from the next iteration

If a SLURM job was in-flight when interrupted:
1. Check job status via `scripts/cluster/status.sh`
2. If still running: wait for completion, then process result
3. If completed: process result, continue loop
4. If failed: revert, log crash, continue loop

## Error Handling

| Error | Recovery |
|-------|----------|
| program.md missing | Prompt user; offer to create from template |
| SSH session expired | Re-establish via ensure_session.sh, retry |
| SLURM submission fails | Log crash, try next hypothesis |
| Git conflict on revert | Hard reset to best_commit, log incident |
| Metric parse failure | Log crash, revert, try next hypothesis |
| Lock acquisition fails | Wait 5s, retry up to 3 times |

## Example Session

```
$ /autoresearch --project ired

Reading program.md... objective: val_mse (minimize)
Reading results.tsv... 0 prior experiments

--- Iteration 1/100 ---
Hypothesis: baseline run (establish starting metric)
Running: sbatch slurm/jobs/autoresearch_pilot.sbatch
Job 62001234 submitted to gpu_test... polling...
Result: val_mse = 0.00977
✓ KEEP: 0.00977 (baseline established)

--- Iteration 2/100 ---
Hypothesis: increase learning rate from 1e-3 to 3e-3
Modified: configs/pilot.json (lr: 1e-3 → 3e-3)
Committed: autoresearch(ired): increase lr from 1e-3 to 3e-3
Running: sbatch slurm/jobs/autoresearch_pilot.sbatch
Job 62001235 submitted... polling...
Result: val_mse = 0.01200
✗ REVERT: 0.01200 (worse than 0.00977)

--- Iteration 3/100 ---
Hypothesis: add weight decay 0.01 (regularize to prevent overfitting)
Modified: configs/pilot.json (weight_decay: 0 → 0.01)
Committed: autoresearch(ired): add weight decay 0.01
Running: sbatch slurm/jobs/autoresearch_pilot.sbatch
Job 62001236 submitted... polling...
Result: val_mse = 0.00950
✓ KEEP: 0.00950 (improved from 0.00977)

...

--- Iteration 47/100 ---
Terminating: plateau detected (15 iterations without improvement)

AUTORESEARCH COMPLETE
Iterations: 47
Best metric: 0.00612 (started at 0.00977)
Improvement: 37.4%
Keeps: 12 / Reverts: 30 / Crashes: 5
Best commit: mno7890
Duration: 6h 23m
```

## Differences from Manual Dispatch

| Aspect | Manual Dispatch | Autoresearch |
|--------|----------------|--------------|
| Human involvement | Every phase transition | None until termination |
| Hypothesis source | Human writes TODO list | Agent generates autonomously |
| Keep/revert decision | Human reviews results | Automatic metric comparison |
| Loop continuation | Manual `/dispatch` calls | Continuous until termination |
| Experiment scope | Full training runs | Short pilot runs (rapid iteration) |
| Code scope | Entire project | Only program.md:files_allowed |
| Decision making | DEBATE phase | No debate — metric decides |

## Related Skills

- **`/dispatch`**: Manual pipeline advancement (autoresearch replaces this)
- **`/make-project`**: Create project with program.md governance
- **`/check-status`**: Monitor autoresearch progress
- **`/ralph-on`**: Enable Ralph loop (autoresearch works with or without Ralph)
