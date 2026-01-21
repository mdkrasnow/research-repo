---
description: Show status across all projects in the research pipeline.
allowed-tools: Read, Glob
---
# /status

Display an overview of all projects in your research pipeline at a glance.

## Usage

```bash
/status                    # Show status of all projects
```

## What It Does

Reads pipeline state from all projects and displays:

- **Project name** and current pipeline phase
- **Next action** (what will happen next)
- **User input status** (if input is needed)
- **Active runs** (currently executing SLURM jobs)

## Output

Summary table showing for each project:

| Project | Phase | Next Action | Status |
|---------|-------|-------------|--------|
| project-1 | IMPLEMENT | Run tests | Ready |
| project-2 | WAIT_SLURM | Poll job 12345 | Running |

## Related Skills

- **`/check-status`**: Get detailed status of all or specific projects
- **`/dispatch`**: Advance projects through their pipeline phases
- **`/check-results`**: Poll results from running SLURM jobs
