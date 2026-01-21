---
description: Main orchestrator: advance up to batch_size projects by one step each and update pipeline/todo/debugging/queue.
allowed-tools: Read, Bash, Task
argument-hint: [--project <slug>]
---
# /dispatch

Advance up to `batch_size` projects (from `.claude/ralph/config.json`) by **one step each**.

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

When launching subagents:
- Provide complete project context in prompt (pipeline.json content, phase, next_action)
- Specify exactly what changes are needed
- Request return of modified file contents or confirmation
- Wait for subagent completion before proceeding
- Incorporate subagent results into orchestration flow

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
