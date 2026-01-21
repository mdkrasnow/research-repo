---
description: Check current state of all projects in pipeline, showing phase, active runs, and next actions.
allowed-tools: Read, Bash
argument-hint: [--project <slug>] [--detailed]
---
# /check-status

Display the current pipeline state across all projects or a specific project.

## Usage

```bash
/check-status                 # Show status of all projects
/check-status --project ired  # Show status of specific project
/check-status --detailed      # Show detailed state including recent events
```

## Output

For each project, displays:
- **Project name** and current phase (INIT/DEBATE/IMPLEMENT/TEST/RUN/CHECK/DEBUG)
- **Next action** (what will happen when dispatched)
- **User input status** (if needs_user_input.value=true, shows the prompt)
- **Active runs** (currently executing SLURM jobs)
- **Recent completed runs** (last 3 completed/failed runs with status and runtime)
- **SLURM polling** (next poll time if waiting on jobs)
- **Last 2 events** (recent actions taken)

## Detailed Mode

With `--detailed` flag, also shows:
- Full event log (all events)
- All completed runs (not just recent)
- Full error details from failed runs
- Queue state from queue.md if available

## Examples

```bash
# Quick status check of all projects
/check-status

# Monitor a specific project
/check-status --project ired

# Deep dive into a project's state
/check-status --project ired-baseline --detailed
```
