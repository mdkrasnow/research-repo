---
description: Show summary of latest dispatch run, what changed, and what's next.
allowed-tools: Read, Bash
argument-hint: [--project <slug>] [--compare-last] [--json]
---
# /dispatch-results

Display a concise summary of the most recent dispatch operations and their outcomes.

## Usage

```bash
/dispatch-results              # Show results from last dispatch run(s)
/dispatch-results --project ired  # Show results for specific project
/dispatch-results --compare-last  # Compare state before/after last dispatch
/dispatch-results --json       # Output as JSON (for scripting)
```

## Output Format

For each project touched in the last dispatch:

```
PROJECT: ired
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status:      Phase transition: RUN → CHECK
Modified:    pipeline.json, documentation/queue.md
Git:         Changes staged but not committed
Active Runs: job_id=55241234 (submitted 2m ago)
Next Poll:   2026-01-21 04:35:00Z (~10m remaining)
Next Action: Poll job status and process results
```

## Options

- `--project <slug>`: Show results for specific project only
- `--compare-last`: Show before/after state (diffs of pipeline.json)
- `--json`: Output as structured JSON (useful for monitoring/alerts)

## What It Shows

For each project touched by dispatch:

- **Phase transition** — Current phase (e.g., RUN → CHECK)
- **Files modified** — Which documentation files were updated
- **Git status** — Staged vs. committed changes
- **Active runs** — Jobs submitted with IDs and timing info
- **Next poll time** — When SLURM status will be checked
- **Next action** — What dispatch will do on next run

## Typical Workflow

```bash
/dispatch                  # Run dispatch
/dispatch-results          # See what changed
/check-status              # Monitor ongoing jobs (optional)
```

## Performance

- Reads only recent pipeline.json changes (fast)
- Compares against git working tree if `--compare-last` used
- Scans up to last 5 projects in pipeline (respects batch_size)
- Typical runtime: <1 second

## Error Handling

- **No recent dispatch**: Returns "No dispatch run found in last 24 hours"
- **Project not found**: Returns "Project <slug> not in recent dispatch results"
- **Invalid JSON output**: Falls back to human-readable format

## Examples

```bash
# Quick check after dispatch
/dispatch-results

# Detailed comparison of what changed
/dispatch-results --compare-last

# Monitoring script (parses JSON)
/dispatch-results --json | jq '.projects[] | select(.phase_changed)'

# Focus on specific project
/dispatch-results --project ired-baseline
```

## Related Skills

- **`/dispatch`**: Run dispatch to advance projects
- **`/check-status`**: Monitor all projects in real-time
- **`/check-results`**: Poll individual SLURM job results
