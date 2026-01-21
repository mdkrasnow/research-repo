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

## Integration with Workflow

Typical sequence:
```bash
/dispatch                  # Run dispatch
/dispatch-results          # See what changed
/check-status              # Monitor ongoing jobs
```

## What It Shows

For each project:
- **Phase transition** (if any occurred)
- **Files modified** (which docs were updated)
- **Git status** (staged vs. committed changes)
- **Active runs** (jobs submitted, with IDs and timing)
- **Next poll time** (when SLURM status will be checked)
- **Next action** (what dispatch will do on next run)

## Performance

- Reads only recent pipeline.json changes (fast)
- Compares against git working tree if `--compare-last` used
- Scans up to last 5 projects in pipeline (respects batch_size)
- Typical runtime: <1 second

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
