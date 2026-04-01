---
description: Run autonomous research loop - hypothesize, code, run, measure, keep/revert, repeat. Karpathy-style autoresearch with parallel candidate tournament.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, Task, Agent
argument-hint: --project <slug> [--iterations N] [--dry-run] [--local]
---
# /autoresearch

Run an autonomous, continuous research loop on a project. The agent forms hypotheses, modifies code, runs experiments, measures results, and keeps or reverts changes — all without human intervention.

Supports **parallel tournament mode**: each round generates N candidate configs, submits all as parallel SLURM jobs, and keeps only the best-performing candidate (if it beats current best).

## Usage

```bash
/autoresearch --project my-experiment              # Run autoresearch loop
/autoresearch --project my-experiment --iterations 20  # Limit to 20 rounds
/autoresearch --project my-experiment --dry-run     # Preview first round only
/autoresearch --project my-experiment --local       # Force local execution (no SLURM)
```

## Prerequisites

The project MUST have a `program.md` governance file at `projects/<slug>/program.md`. This file defines:
- **Objective metric** and how to measure it
- **Constraints** (files allowed, time budgets)
- **Ratchet rules** (when to keep vs revert)
- **Termination conditions**
- **parallel_candidates** (number of parallel experiments per round, default 1 for sequential mode)

If `program.md` doesn't exist, prompt the user to create one (offer to generate from template at `templates/program.md`).

## The Parallel Autoresearch Loop

```
┌──────────────────────────────────────────────────────┐
│  AUTORESEARCH LOOP (runs until termination)          │
│                                                      │
│  1. READ CONTEXT                                     │
│     ├─ program.md (governance)                       │
│     ├─ results.tsv (experiment history)              │
│     ├─ Current code state (baseline.json)            │
│     └─ pipeline.json (autoresearch state)            │
│                                                      │
│  2. HYPOTHESIZE (generate N candidates)              │
│     ├─ Analyze past results                          │
│     ├─ Identify N promising directions               │
│     ├─ Each candidate: ONE change from current best  │
│     └─ Candidates should explore DIFFERENT dimensions│
│                                                      │
│  3. IMPLEMENT                                        │
│     ├─ For each candidate i (1..N):                  │
│     │   ├─ Create configs/candidate_i.json           │
│     │   └─ Create configs/eval_candidate_i.json      │
│     ├─ Modify ONLY files in program.md:files_allowed │
│     └─ Git commit all candidates together            │
│                                                      │
│  4. RUN (parallel)                                   │
│     ├─ Submit N SLURM jobs simultaneously            │
│     │   (each with its own CONFIG_PATH/EVAL_CONFIG)  │
│     ├─ Use partition diversification for 4+ jobs     │
│     └─ Poll all until complete                       │
│                                                      │
│  5. MEASURE                                          │
│     ├─ Parse metric from each candidate's eval log   │
│     └─ Rank all N candidates by metric               │
│                                                      │
│  6. TOURNAMENT RATCHET                               │
│     ├─ Find best candidate among N                   │
│     ├─ If best beats current best_so_far:            │
│     │   ├─ WINNER: merge its config into baseline    │
│     │   ├─ Update best_so_far, best_commit           │
│     │   ├─ Other candidates marked SKIP              │
│     │   └─ Reset consecutive_failures to 0           │
│     ├─ If NO candidate beats current best:           │
│     │   ├─ All marked SKIP                           │
│     │   ├─ Revert the candidates commit              │
│     │   └─ Increment consecutive_failures            │
│     └─ Clean up candidate config files               │
│                                                      │
│  7. LOG                                              │
│     ├─ Append ALL N candidates to results.tsv        │
│     └─ Update pipeline.json                          │
│                                                      │
│  8. CHECK TERMINATION                                │
│     ├─ Max rounds reached?                           │
│     ├─ Target metric achieved?                       │
│     ├─ Plateau detected?                             │
│     ├─ Max wall hours exceeded?                      │
│     └─ Max consecutive round failures?               │
│                                                      │
│  → If not terminated: go to step 1                   │
└──────────────────────────────────────────────────────┘
```

## Detailed Step Instructions

### Step 1: Read Context

```
1. Read projects/<slug>/program.md
   - Parse all governance fields
   - Read parallel_candidates (default 1 = sequential mode)

2. Read projects/<slug>/results.tsv (create if missing)
   - Header: round | candidate | metric | status | description | git_sha | timestamp
   - Parse all past results to understand what's been tried

3. Read current state of baseline config (configs/baseline.json)
   - This is the "current best" config that candidates modify from

4. Read projects/<slug>/.state/pipeline.json
   - Check phase (should be AUTORESEARCH or will be set to it)
   - Check autoresearch.round for where we left off
```

### Step 2: Hypothesize (N candidates)

**CRITICAL**: The agent must form its OWN hypotheses. Do NOT ask the user what to try.

```
Based on results.tsv and program.md:exploration_dimensions:

1. If results.tsv is empty (first round):
   - Candidate 1: Run baseline as-is (establishes starting metric)
   - Remaining candidates: simple single-dimension variations
     (e.g., different lr, different gamma, different epsilon)

2. If results.tsv has entries:
   - Analyze which dimensions have been explored, what worked/didn't
   - Generate N diverse hypotheses across DIFFERENT dimensions
   - Diversity rule: no two candidates in same round should change the same dimension
   - Each hypothesis: ONE change from current baseline.json

3. Hypothesis quality rules:
   - ONE change per candidate (isolable)
   - Motivated by past results (not random)
   - Specific and testable ("increase lr from 1e-3 to 3e-3")
   - Not a repeat of a previously failed approach
   - N candidates should span different exploration_dimensions
```

### Step 3: Implement (create candidate configs)

```
For each candidate i (1..N):

1. Copy baseline.json → configs/candidate_i.json
   - Apply the ONE change for this hypothesis
   - Set output_dir to "projects/<slug>/results/candidate_i"
   - Set experiment_name and description

2. Create configs/eval_candidate_i.json
   - Copy from configs/eval.json
   - Set checkpoint_path to "projects/<slug>/results/candidate_i/final_model.pt"

3. After all N configs created:
   - Git commit: "autoresearch(<slug>): round R — N candidates: <brief list>"

IMPORTANT:
- ONLY modify/create files in program.md:files_allowed (configs/*.json qualifies)
- If code changes are needed (train_dganm.py), apply them BEFORE creating configs
- Code changes apply to ALL candidates in the round
```

### Step 4: Run (parallel SLURM)

```
1. Ensure SSH session: scripts/cluster/ensure_session.sh

2. For each candidate i (1..N), submit a SLURM job:
   - CONFIG_PATH=projects/<slug>/configs/candidate_i.json
   - EVAL_CONFIG=projects/<slug>/configs/eval_candidate_i.json
   - Use partition diversification for 4+ jobs:
     - First half → gpu_test
     - Second half → gpu

3. Record all job_ids in pipeline.json:active_runs

4. Poll all jobs until complete:
   - First poll at 60s (catch init errors)
   - Then every 2 minutes
   - A round is done when ALL N jobs are complete
   - If any job crashes, mark it as CRASH and continue waiting for others

5. Fetch logs for ALL completed jobs:
   scripts/cluster/remote_fetch.sh <slug>
```

### Step 5: Measure

```
For each candidate i:

1. Find eval log: slurm/logs/eval_<job_id>.log
   - Parse for eval_grep pattern (e.g., "^short_horizon_recovery_distance:")
   - Extract metric value

2. Handle edge cases:
   - Metric not found → candidate status: CRASH, metric: null
   - NaN or Inf → candidate status: CRASH, metric: null
   - Job failed → candidate status: CRASH, metric: null

3. Build results table:
   candidate_results = [
     {candidate: 1, metric: 0.042, description: "increase lr"},
     {candidate: 2, metric: 0.044, description: "add wd"},
     {candidate: 3, metric: null, description: "reduce depth (CRASHED)"},
     {candidate: 4, metric: 0.041, description: "increase epsilon"},
     {candidate: 5, metric: 0.045, description: "double mining_steps"},
   ]
```

### Step 6: Tournament Ratchet

```
1. Filter out crashed candidates (metric = null)

2. Find best candidate:
   - direction=minimize: candidate with lowest metric
   - direction=maximize: candidate with highest metric

3. Compare best candidate to current best_so_far:

   IF best candidate beats best_so_far (or this is round 1 establishing baseline):
     a. Winner's config → merge into baseline.json
        - Read candidate_i.json, copy all training params to baseline.json
        - Reset output_dir to standard path
     b. Update program.md: best_so_far, best_commit
     c. Update eval.json: checkpoint_path points to standard output
     d. Log winner as WINNER in results.tsv
     e. Log others as SKIP in results.tsv
     f. Reset consecutive_failures = 0
     g. Git commit: "autoresearch(<slug>): round R WINNER — <description> (metric)"
     h. Clean up candidate configs
     i. **FID evaluation trigger** (if configured): Submit async FID eval job
        - See "FID Evaluation Integration" section below

   IF no candidate beats best_so_far:
     a. Revert the candidates commit: git revert HEAD --no-edit
     b. Log all as SKIP in results.tsv
     c. Increment consecutive_failures
     d. Print: "Round R: no improvement. Best remains <best_so_far>"
     e. Clean up candidate configs (they're reverted anyway)
```

**Status values in results.tsv:**
- `WINNER` — best candidate that beat best_so_far (merged into baseline)
- `SKIP` — non-winning candidate (discarded)
- `CRASH` — candidate that failed to produce a metric
- `BASELINE` — first-round baseline establishment

### Step 7: Log Results

```
1. Append ALL N candidates to projects/<slug>/results.tsv:
   round | candidate | metric | status | description | git_sha | timestamp

   Example:
   2   1   0.042   SKIP     increase lr to 3e-4          abc123   2026-04-01T10:00:00Z
   2   2   0.041   WINNER   increase epsilon to 0.2      abc123   2026-04-01T10:00:00Z
   2   3   null    CRASH    reduce depth to 8 (OOM)      abc123   2026-04-01T10:00:00Z
   2   4   0.044   SKIP     double mining_steps to 6     abc123   2026-04-01T10:00:00Z
   2   5   0.045   SKIP     add weight decay 0.01        abc123   2026-04-01T10:00:00Z

2. Update projects/<slug>/.state/pipeline.json:
   - autoresearch.round: <current>
   - autoresearch.best_metric: <best_so_far>
   - autoresearch.best_commit: <sha>
   - autoresearch.consecutive_failures: <count>
   - autoresearch.total_rounds: <count>
   - autoresearch.total_candidates: <total experiments run>
   - autoresearch.total_winners: <count>

3. Print round summary:
   "Round <R>/<max>: Best candidate: <metric> (<status>)
    Candidates: <c1_metric> | <c2_metric> | ... | <cN_metric>
    Overall best: <best_so_far> | Winners: <w> / Rounds: <r>"
```

### Step 8: Check Termination

```
Check each condition in order:

1. max_iterations reached? (counts ROUNDS, not individual candidates)
   → "Terminating: max rounds (<N>) reached. Best: <metric> at <commit>"

2. target_metric achieved?
   → "Terminating: target metric (<target>) achieved! Final: <metric>"

3. max_consecutive_failures reached? (rounds with no winner)
   → "Terminating: <N> consecutive rounds with no improvement."

4. plateau detected? (stop_on_plateau && no improvement in plateau_window rounds)
   → "Terminating: plateau detected."

5. max_wall_hours exceeded?
   → "Terminating: wall-clock limit exceeded."

If none triggered: CONTINUE to next round.

On termination:
  1. Update pipeline.json: phase → "CHECK"
  2. Update pipeline.json: autoresearch.terminated_reason = <reason>
  3. Print final summary with total rounds, total candidates, best metric, improvement %
  4. Commit results.tsv
```

## Candidate Config Convention

Training configs: `configs/candidate_1.json` through `configs/candidate_N.json`
Eval configs: `configs/eval_candidate_1.json` through `configs/eval_candidate_N.json`

Each candidate_i.json is a modified copy of baseline.json with:
- Unique `experiment_name`: "round_R_candidate_i_<brief>"
- Unique `output_dir`: "projects/<slug>/results/candidate_i"
- Descriptive `description`
- The ONE parameter change being tested

Each eval_candidate_i.json is a copy of eval.json with:
- `checkpoint_path`: "projects/<slug>/results/candidate_i/final_model.pt"

After a round completes:
- Winner's training params merged into baseline.json
- All candidate_*.json and eval_candidate_*.json cleaned up
- eval.json updated if winner changed the checkpoint path

## Pipeline.json Schema Extension

```json
{
  "phase": "AUTORESEARCH",
  "autoresearch": {
    "active": true,
    "round": 5,
    "parallel_candidates": 5,
    "best_metric": 0.00823,
    "best_commit": "abc1234",
    "baseline_metric": 0.00977,
    "consecutive_failures": 1,
    "total_rounds": 5,
    "total_candidates": 25,
    "total_winners": 3,
    "total_crashes": 2,
    "started_at": "2026-03-31T10:00:00Z",
    "last_round_at": "2026-03-31T14:30:00Z",
    "terminated_reason": null
  }
}
```

## Results.tsv Format

Tab-separated file at `projects/<slug>/results.tsv`:

```
round	candidate	metric	status	description	git_sha	timestamp
1	1	0.04524	BASELINE	3-epoch baseline (no mining)	abc123	2026-04-01T10:00:00Z
2	1	0.04400	SKIP	increase lr to 3e-4	def456	2026-04-01T10:30:00Z
2	2	0.04100	WINNER	enable mining, gamma=0.5, margin=5.0	def456	2026-04-01T10:30:00Z
2	3	0.04500	SKIP	add weight decay 0.01	def456	2026-04-01T10:30:00Z
2	4	null	CRASH	reduce depth to 8 (OOM)	def456	2026-04-01T10:30:00Z
2	5	0.04450	SKIP	increase batch_size to 256	def456	2026-04-01T10:30:00Z
```

## Partition Diversification for Parallel Jobs

When submitting 4+ parallel candidates:
- Split jobs across gpu_test and gpu partitions
- Candidates 1..ceil(N/2) → gpu_test (higher priority)
- Candidates ceil(N/2)+1..N → gpu (standard priority)
- This avoids QOS per-partition submission limits

## Integration with Existing System

- **Autoresearch IS a dispatch mode**: When `phase=AUTORESEARCH`, `/dispatch` delegates to the autoresearch loop
- **Ralph loop compatible**: Ralph's stop hook recognizes AUTORESEARCH phase as actionable
- **Lock-aware**: Each round acquires/releases the project lock
- **Git-disciplined**: Every change is committed; reverts are clean revert commits
- **SLURM-aware**: Uses existing `scripts/cluster/*` infrastructure
- **Results persist**: results.tsv and pipeline.json survive across sessions

## Resuming Autoresearch

If a session ends mid-autoresearch:

1. `/autoresearch --project <slug>` reads pipeline.json
2. Sees `phase=AUTORESEARCH` with `autoresearch.active=true`
3. Reads `autoresearch.round` to know where it left off
4. Checks `active_runs` — if jobs were in-flight, checks their status
5. If jobs still running: wait for completion, then process results
6. If jobs completed: process results, continue to next round
7. If no active jobs: start next round

## Backward Compatibility (Sequential Mode)

If `parallel_candidates: 1` (or not set) in program.md, the loop behaves exactly like the original sequential autoresearch:
- 1 hypothesis per round
- 1 SLURM job
- KEEP/REVERT decision (no tournament)
- results.tsv uses candidate=1 for all rows

## Error Handling

| Error | Recovery |
|-------|----------|
| program.md missing | Prompt user; offer to create from template |
| SSH session expired | Re-establish via ensure_session.sh, retry |
| All N candidates crash | Log crashes, increment failures, try next round |
| Some candidates crash | Exclude crashed from tournament, proceed with survivors |
| Git conflict on revert | Hard reset to best_commit, log incident |
| Metric parse failure | Mark candidate as CRASH, exclude from tournament |
| QOS limit hit | Use partition diversification, retry failed submissions |

## FID Evaluation Integration

When a project has a FID evaluation config (`configs/eval_fid.json`), autoresearch automatically submits an async FID evaluation job after each WINNER is declared. This runs concurrently with the next round — it does NOT block the proxy-metric loop.

### Trigger Conditions

FID eval is triggered when ALL of these are true:
1. `configs/eval_fid.json` exists in the project
2. A WINNER was declared this round (not on SKIP-all rounds)
3. `slurm/jobs/fid_eval.sbatch` exists

### Steps (after Step 6g, as Step 6i)

```
1. Create a round-specific FID eval config:
   - Copy configs/eval_fid.json → /tmp config (or use env vars)
   - Set checkpoint_path to winner's checkpoint path
   - Set save_samples_path to "projects/<slug>/results/fid_samples_round_R.png"

2. Submit FID eval job (fire-and-forget, async):
   FID_CONFIG=projects/<slug>/configs/eval_fid.json \
   GIT_SHA=<winner_commit> \
     scripts/cluster/submit.sh projects/<slug>/slurm/jobs/fid_eval.sbatch

3. Log the FID job submission:
   - Print: "FID eval submitted for round R winner (job <id>)"
   - Add to pipeline.json:autoresearch.fid_jobs array:
     {"round": R, "job_id": "<id>", "status": "submitted", "submitted_at": "<ts>"}

4. Do NOT wait for FID results — proceed immediately to next round.
```

### FID Results Collection

FID results are collected opportunistically:
- At the START of each round (Step 1), check if any pending FID jobs have completed
- Parse `fid:` line from `slurm/logs/fid_eval_<job_id>.log`
- Update pipeline.json:autoresearch.fid_jobs entry with `fid_score` and `status: "completed"`
- Append to `projects/<slug>/fid_results.tsv`:
  ```
  round	fid	proxy_metric	checkpoint	job_id	timestamp
  4	142.3	0.010786	results/baseline/final_model.pt	12345	2026-04-01T05:00:00Z
  ```
- Print: "FID update: round R = <fid> (proxy was <proxy_metric>)"

### FID Results File

`projects/<slug>/fid_results.tsv` tracks FID scores over time:
```
round	fid	proxy_metric	checkpoint	job_id	timestamp
1	285.4	0.045237	results/baseline/final_model.pt	2861800	2026-03-31T20:00:00Z
2	198.7	0.016082	results/baseline/final_model.pt	2893600	2026-04-01T00:00:00Z
3	172.1	0.013018	results/baseline/final_model.pt	2901234	2026-04-01T02:00:00Z
```

This lets you track whether proxy metric improvements correlate with FID improvements.

### Divergence Alert

If FID WORSENS by >20% while proxy metric improved, print a warning:
```
⚠ FID DIVERGENCE: round R proxy improved (0.013→0.011, -15%) but FID worsened (172→215, +25%)
Consider: proxy metric may not reflect generative quality. Review sample_grid.png.
```

## Related Skills

- **`/dispatch`**: Manual pipeline advancement (autoresearch replaces this)
- **`/make-project`**: Create project with program.md governance
- **`/check-status`**: Monitor autoresearch progress
- **`/ralph-on`**: Enable Ralph loop (autoresearch works with or without Ralph)
