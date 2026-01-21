---
description: Main orchestrator: advance up to batch_size projects by one step each and update pipeline/todo/debugging/queue.
allowed-tools: Read, Bash, Task
argument-hint: [--project <slug>] [--dry-run] [--verbose]
---
# /dispatch

Advance up to `batch_size` projects (from `.claude/ralph/config.json`) by **one step each**.

## Quick Start

```bash
/check-status           # See current state of all projects
/ensure-session         # Verify cluster access
/dispatch --dry-run     # Preview what dispatch will do
/dispatch               # Execute dispatch actions
```

## Options

- `--project <slug>`: Only dispatch single project (ignores batch_size)
- `--dry-run`: Preview actions without executing (shows plan, doesn't modify files)
- `--verbose`: Show detailed subagent prompts and communication

## Architecture: Subagent-Driven Code Execution

**CRITICAL**: This orchestrator MUST ALWAYS use the Task tool with specialized subagents for any code modifications. The main agent is a coordinator only.

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

### Subagent Communication

When launching subagents, use this context template:

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
2. Confirm lock is still held by main orchestrator (do NOT release)
3. List any validation issues or warnings
```

Best practices:
- Always include full pipeline.json content (gives subagent complete context)
- Include git SHA (important for experiment reproducibility)
- Specify exact files to modify (prevents unintended changes)
- Request explicit confirmation of changes (safety check)
- Subagent must NOT release the lock (main orchestrator handles this)

## Dry-Run Mode (`--dry-run`)

When `--dry-run` is enabled:

1. **Skip all file modifications** - No changes to pipeline.json, docs, or git
2. **Skip all SLURM operations** - No job submissions, cancellations, or status checks
3. **Show dispatch plan**:
   - List all projects that would be processed (up to batch_size)
   - For each project:
     - Current phase and next_action
     - What will be executed (e.g., "Dispatch subagent to analyze logs and update pipeline")
     - Estimated impact (files modified, phase transitions)
     - Any blockers (e.g., "Blocked: needs_user_input=true")
4. **No git operations** - Subagents are not launched, nothing is committed

Typical workflow:
```bash
/check-status --detailed    # Understand current state
/dispatch --dry-run         # See what will happen
# Review the plan...
/dispatch                   # Execute if plan looks good
```

Cluster-aware rule:
- **CRITICAL**: SLURM is NEVER available locally. ALL SLURM operations MUST use remote execution via `scripts/cluster/*`.
- If `phase` implies SLURM actions (RUN / WAIT_SLURM / CHECK), use remote execution via `scripts/cluster/*`.
- If no SSH session exists (checked via `scripts/cluster/ensure_session.sh`), set `needs_user_input=true` with prompt:
  "Run scripts/cluster/ssh_bootstrap.sh (interactive SSH+2FA) then rerun /dispatch"
  and DO NOT spin.

## Job Verification (MUST DO FIRST)
Before processing any projects, verify all outstanding SLURM jobs:
1. For each project with `active_runs` in pipeline.json:
   - Use Bash to check if jobs are running: `scripts/cluster/status.sh <job_id>`
   - If job is NOT running but status says "running":
     - Use Bash to fetch logs: `scripts/cluster/remote_fetch.sh <project_slug>`
     - Logs will be in `slurm/logs/<job_name>_<job_id>.{out,err}`
     - **Dispatch a subagent** to analyze logs and update pipeline:
       - Subagent reads log files and determines failure reason
       - Subagent adds detailed entry to `documentation/debugging.md` with:
         - Error message from logs
         - Root cause analysis
         - Run ID and job ID
         - Log file paths
         - Timestamp
       - Subagent updates `pipeline.json`:
         - Moves job from `active_runs` to `completed_runs` with `status: "failed"`
         - Adds `failed_at` timestamp
         - Adds `error` field with failure summary
       - Subagent sets `phase=DEBUG` if blocking, otherwise continues
2. Only after all jobs verified and subagent updates complete, proceed with normal dispatch

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

4. **Orchestrate Based on Phase**:
   - DEBUG: Subagent analyzes logs, updates pipeline with diagnosis
   - IMPLEMENT: Subagent creates code changes per specification
   - TEST: Subagent runs tests, documents results, updates pipeline
   - RUN: Use Bash directly to run `scripts/cluster/remote_submit.sh`
   - CHECK/POLL: Use Bash to check status via `scripts/cluster/status.sh`
   - DEBATE: Subagent conducts analysis, documents decision

5. **Release Lock**:
   - `scripts/ralph/lock.sh release projects/<slug>`

### Main Orchestrator Responsibilities

The main orchestrator:
- ✓ Reads project state (Read tool only)
- ✓ Coordinates subagent execution via Task tool
- ✓ Runs cluster commands via Bash (scripts/cluster/*)
- ✗ Does NOT modify code directly
- ✗ Does NOT perform code searches (delegates to Explore subagent)
- ✗ Does NOT create/edit project files

Hybrid SLURM waiting:
- If a job is RUNNING and `next_poll_after` is in the future, skip it and work on other projects.
- If nothing actionable, exit (Stop hook will then allow stopping).

## Error Handling & Common Scenarios

### SSH Session Missing
**Symptom**: dispatch fails with "SSH session not configured"
**Recovery**:
```bash
/ensure-session --init     # Configure or reconfigure cluster access
/dispatch --dry-run        # Preview again
/dispatch                  # Proceed
```

### Lock Acquisition Failure
**Symptom**: "Failed to acquire lock for projects/ired"
**Cause**: Another dispatch is running or lock file is stale
**Recovery**:
```bash
scripts/ralph/lock.sh status projects/ired  # Check lock status
scripts/ralph/lock.sh release projects/ired # Force release if stale (CAREFUL!)
/dispatch                                   # Retry
```

### Subagent Failure
**Symptom**: Subagent reports error during pipeline.json update
**Recovery**:
```bash
/check-status --project <slug>  # Verify current state
# Manually fix the issue if needed
/dispatch --dry-run             # Preview retry
/dispatch                       # Retry dispatch
```

### SLURM Job Status Uncertain
**Symptom**: Job shows "running" but can't verify on cluster
**Recovery**:
```bash
/ensure-session --verify        # Verify cluster connectivity
/dispatch --verbose             # Show detailed status checks
```

## Troubleshooting

1. **Always run `/check-status` first** to understand current state
2. **Use `--dry-run` before executing** to preview changes
3. **Use `--verbose` to see subagent context** and debug communication
4. **Check `CLAUDE.md`** for authoritative dispatch rules and constraints
5. **Review recent events** in pipeline.json to understand context

## Integration with Other Skills

### Recommended Workflow

```bash
# Step 1: Check what's happening
/check-status

# Step 2: Ensure cluster is ready
/ensure-session

# Step 3: Preview dispatch actions
/dispatch --dry-run

# Step 4: Review the plan
# ... manually review output

# Step 5: Execute dispatch
/dispatch

# Step 6: Monitor progress
/check-status --detailed

# Repeat as needed based on phase transitions
```

### Skill Relationships

- **`/check-status`**: Use before `/dispatch` to understand current state
- **`/ensure-session`**: Use before any dispatch that involves SLURM operations
- **`/dispatch --dry-run`**: Safe preview of all planned changes
- **`/status`** (system-wide): Shows overall project pipeline state

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
