# research-repo

A multi-project research repository template designed for **Claude Code** with an autonomous **Ralph Wiggum** loop.

## What you get
- A consistent per-project layout (`projects/<slug>/...`) with isolated outputs and state
- Global research docs (SLURM, experimental design, reproducibility)
- Claude Code custom slash commands in `.claude/commands/`
- A **Stop hook** that implements a **classic Ralph Wiggum loop** using **exit code 2** (blocks stopping and feeds guidance to Claude)
- A sample project demonstrating the flow end-to-end

## Quick start (Claude Code)
1. Open this repo in Claude Code.
2. Run:
   - `/ralph-on` to enable the loop
   - `/dispatch` to start the orchestrator
3. Claude will iterate until no actionable work remains (or until max iterations).

To disable:
- `/ralph-off`

## Projects
See `projects/PROJECTS.md`.

Generated: 2026-01-12T14:32:10Z
