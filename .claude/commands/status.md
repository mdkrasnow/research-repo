---
description: Show status across all projects.
allowed-tools: Read, Glob
---
# /status
Read all `projects/*/.state/pipeline.json` and summarize:
- phase
- next_action
- needs_user_input
- active_runs
