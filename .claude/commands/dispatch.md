---
description: Main orchestrator for advancing projects through their pipeline phases.
allowed-tools: Read, Bash, Task
argument-hint: [--project <slug>] [--dry-run] [--verbose]
---
# /dispatch

Advance up to `batch_size` projects by one step each, orchestrating all code modifications and state updates.

## Usage

```bash
/dispatch                          # Process up to batch_size projects
/dispatch --project my-experiment  # Process single project
/dispatch --dry-run                # Preview changes without executing
/dispatch --verbose                # Show detailed orchestration logs
```

## Quick Start Workflow

```bash
/check-status           # ① See current state of all projects
/ensure-session         # ② Verify cluster access (if needed)
/dispatch --dry-run     # ③ Preview what will happen
/dispatch               # ④ Execute dispatch
/dispatch-results       # ⑤ See what changed
```

## Options

| Flag | Purpose |
|------|---------|
| `--project <slug>` | Process single project (ignores batch_size) |
| `--dry-run` | Preview without modifying files or submitting jobs |
| `--verbose` | Show detailed subagent communication and context |

## Architecture: Subagent-Driven Execution

**CRITICAL**: Dispatch delegates ALL code modifications to specialized subagents. Main orchestrator is coordinator only.

### How Dispatch Works

```
Main Orchestrator
├── ① Verify SLURM jobs (Bash)
├── ② Acquire project lock (Bash)
├── ③ Read pipeline state (Read)
├── ④ Delegate modifications → Task(Subagent)
├── ⑤ Execute cluster operations (Bash + scripts/cluster/*)
└── ⑥ Release project lock (Bash)
```

### Delegation Rules

1. **Code Changes** (all modifications to project files):
   - Use `Task` tool with `subagent_type=Explore` or specialized agent
   - Include full context: project slug, pipeline state, required changes, git information
   - Do NOT attempt Write/Edit/Edit directly in main orchestrator
   - Examples:
     - Modifying sbatch scripts → use Task with general-purpose agent
     - Updating pipeline.json → use Task with general-purpose agent
     - Creating run folders → use Task with general-purpose agent

2. **Information Gathering**:
   - Use Task with `subagent_type=Explore` for codebase exploration
   - Use Task with `subagent_type=general-purpose` for complex searches
   - Main agent may use Read/Bash for quick checks only

3. **Git Operations**:
   - Use Bash directly for git status/log/fetch/pull
   - Use Bash for git add/commit/push (after subagent makes changes)
   - Use Bash for branch management

4. **Cluster Operations**:
   - Use Bash with `scripts/cluster/*` for remote SLURM operations
   - Use Bash for ssh.sh, status.sh, submit.sh, etc.
   - Do NOT attempt to manage cluster state in main orchestrator

### Subagent Context Template

Launch subagents with this structured context:

```
PROJECT: <slug>
PHASE: <current_phase>
GIT_SHA: <commit_hash>
LOCK: projects/<slug> (acquired by main orchestrator)

PIPELINE STATE:
<full pipeline.json content>

REQUIRED CHANGES:
1. <specific change 1>
2. <specific change 2>

FILES TO MODIFY:
- .state/pipeline.json
- documentation/debugging.md
- [other files if applicable]

AFTER COMPLETING CHANGES:
1. Return summary of modifications
2. Confirm lock is still held
3. List any warnings or validation issues
```

**Critical Best Practices:**

- ✓ Include full pipeline.json (gives subagent complete context)
- ✓ Include git SHA (for reproducibility)
- ✓ Specify exact files to modify (prevents unintended changes)
- ✓ Request explicit confirmation of changes
- ✗ Subagent must NOT release lock (main orchestrator does this)

## Dry-Run Mode

Preview all dispatch actions without making any changes:

```bash
/dispatch --dry-run     # See what WOULD happen
```

**What Dry-Run Does:**

| Operation | Behavior |
|-----------|----------|
| File modifications | **Skipped** — No changes to pipeline.json, docs |
| SLURM operations | **Skipped** — No jobs submitted/cancelled |
| Subagent launches | **Skipped** — No code executed |
| Git operations | **Skipped** — No commits |
| Plan output | **Shown** — Full preview of what WOULD run |

**Dry-Run Output:**

```
DISPATCH PLAN (DRY-RUN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project: my-experiment
Current Phase: IMPLEMENT
Action: Run implementation tasks
Files Modified: pipeline.json, implementation-todo.md
Next Phase: TEST (if all tasks complete)

Project: baseline
Current Phase: WAIT_SLURM
Action: Poll job status
Job ID: 55241234
Next Poll: In 5 minutes
```

**Typical Workflow:**

```bash
/check-status --detailed    # ① Understand current state
/dispatch --dry-run         # ② Preview what will happen
# Review the output...
/dispatch                   # ③ Execute if plan looks good
/dispatch-results           # ④ See what changed
```

## Remote SLURM Execution

**CRITICAL**: SLURM is NEVER available locally. ALL operations use remote execution via `scripts/cluster/*`.

Remote execution rules:

- If `phase` implies SLURM actions (RUN / WAIT_SLURM / CHECK), use `scripts/cluster/*`
- If SSH session missing, set `needs_user_input=true` with bootstrap prompt
- Return without executing dispatch; user must bootstrap first

## Job Verification (Step 1)

Before processing any projects, verify all outstanding SLURM jobs:

```
For each project with active_runs:
  1. Check if job is actually running via scripts/cluster/status.sh
  2. If NOT running but marked "running":
     a. Fetch logs via scripts/cluster/remote_fetch.sh
     b. Dispatch subagent to analyze and diagnose
     c. Update debugging.md with error details
     d. Move to completed_runs with status="failed"
  3. If running: skip to next project
```

**Subagent Responsibilities (on failed job):**

- Parse SLURM logs to determine failure reason
- Write detailed entry to `documentation/debugging.md`
- Update `pipeline.json`:
  - Move from `active_runs` → `completed_runs` with `status: "failed"`
  - Add `failed_at` timestamp
  - Add `error` summary
- Set `phase=DEBUG` if blocking, else continue

**Only after all jobs verified**, proceed with normal dispatch.

## Dispatch Execution (Step 2)

For each touched project:

1. **Acquire Lock**:
   - `scripts/ralph/lock.sh acquire projects/<slug>`

2. **Read Current State**:
   - Read pipeline.json to determine phase and next_action
   - Understand what modifications are needed

3. **Dispatch Subagent for Code Changes**:
   - Use `Task` tool to launch a general-purpose subagent
   - Provide complete project context in prompt
   - Subagent handles all file modifications:
     - `.state/pipeline.json` updates
     - `documentation/implementation-todo.md` updates
     - `documentation/debugging.md` updates
     - `documentation/queue.md` updates
     - run ledger updates under `runs/<run_id>/`
   - Wait for subagent completion

4. **Execute Based on Phase**:

| Phase | Action | Tool |
|-------|--------|------|
| DEBUG | Analyze failed logs, diagnose issue | Subagent |
| IMPLEMENT | Create code changes per specification | Subagent |
| TEST | Run tests, document results | Subagent |
| RUN | Submit job to SLURM cluster | Bash + scripts/cluster/* |
| CHECK/POLL | Check job status remotely | Bash + scripts/cluster/* |
| DEBATE | Conduct structured debate | Subagent |

5. **Release Lock**:
   - `scripts/ralph/lock.sh release projects/<slug>`

### Main Orchestrator Responsibilities

**What Main Orchestrator Does:**

- ✓ Read project state via Read tool
- ✓ Coordinate subagent execution via Task tool
- ✓ Run cluster commands via Bash (scripts/cluster/*)
- ✓ Manage locks (acquire/release)
- ✓ Verify SLURM job status
- ✓ Make phase transition decisions

**What Main Orchestrator Does NOT Do:**

- ✗ Modify code directly (delegates to subagents)
- ✗ Perform code searches (delegates to Explore subagent)
- ✗ Create/edit project files (subagents do this)

### Hybrid SLURM Waiting

**Intelligent job polling strategy:**

- If job is RUNNING and `next_poll_after` in future → skip, work on other projects
- If job is RUNNING and poll time due → check status, update polling schedule
- If nothing actionable across all projects → exit dispatch (Stop hook allows stopping)
- If some projects blocked on user input → exit dispatch

## Error Handling

| Symptom | Cause | Recovery |
|---------|-------|----------|
| "SSH session not configured" | Cluster access not configured | `/ensure-session --init`, then retry |
| "Failed to acquire lock" | Another dispatch running or stale lock | Check with `scripts/ralph/lock.sh status`, force release if stale |
| "Subagent failed" | Error during pipeline update | `/check-status`, fix issue, `--dry-run`, retry |
| "SLURM status uncertain" | Cluster connectivity issue | `/ensure-session --verify`, `--verbose` for details |
| "No active projects" | All projects completed or blocked | Check with `/status` for overall state |

## Troubleshooting Workflow

**When dispatch isn't working as expected:**

1. **Check current state**: `/check-status --detailed`
2. **Preview actions**: `/dispatch --dry-run`
3. **Enable diagnostics**: `/dispatch --verbose`
4. **Verify cluster**: `/ensure-session --verify`
5. **Review rules**: Check `CLAUDE.md` for authoritative constraints
6. **Debug specific project**: `/check-status --project <slug>`

## Related Skills

Dispatch works within an ecosystem of complementary skills:

| Skill | When to Use |
|-------|-------------|
| **`/check-status`** | Before dispatch — understand current state |
| **`/status`** | Quick system-wide overview of all projects |
| **`/ensure-session`** | Verify cluster connectivity before RUN/CHECK phases |
| **`/dispatch-results`** | After dispatch — see what changed |
| **`/check-results`** | Poll individual SLURM job results |
| **`/run-experiments`** | Manual job submission (dispatch calls this) |
| **`/parallel-implement`** | Parallelize tasks within single project |
| **`/make-project`** | Create new project (dispatch advances it) |
| **`/debate`** | Structured decision-making (dispatch calls this) |

## Typical Workflow

```bash
/check-status                   # ① Understand current state
/ensure-session                 # ② Verify cluster access
/dispatch --dry-run             # ③ Preview all actions
# [review the plan]
/dispatch                       # ④ Execute dispatch
/dispatch-results               # ⑤ See what changed
# [wait for jobs to complete]
/check-results --project <slug> # ⑥ Poll job status
# [repeat as needed]
```

## Performance Notes

- Dispatch processes up to `batch_size` projects per invocation (configurable in `.claude/ralph/config.json`)
- SLURM job status checks are batched (one per project with active_runs)
- Subagent context includes full pipeline.json (typically 300-400 lines)
- Lock acquisition timeout is 30 seconds per project
- Overall dispatch runtime typically 1-5 minutes depending on SLURM status checks

## Implementation Notes for Developers

### Subagent Return Values
After executing a subagent, expect:
- Summary of files modified
- Confirmation of lock status
- Any validation warnings or errors
- New phase (if phase transition occurred)
- List of `active_runs` or `completed_runs` added/modified

### Lock Management
- Main orchestrator acquires lock before dispatch step
- Subagent must NOT release lock (main orchestrator releases)
- If subagent fails, main orchestrator still releases lock
- Lock protects all file modifications for a project
