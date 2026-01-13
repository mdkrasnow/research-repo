# System Spec (One Page)

**Goal:** autonomously advance multiple research projects in Claude Code.

**Key mechanisms**
- Global commands in `.claude/commands/`
- Per-project state in `projects/<slug>/.state/pipeline.json`
- Stop hook loop (“Ralph Wiggum classic”): blocks Stop with exit code 2 while work is actionable

**Dispatch precedence**
1) DEBUG blockers
2) CHECK/POLL completed jobs
3) TEST
4) RUN
5) IMPLEMENT
6) DEBATE
7) IDLE

**Hybrid SLURM waiting**
- Poll with backoff; if not due, work on other projects
- If nothing actionable, allow Stop (prevents hot loop)

**Required outputs**
- Update pipeline.json after every step
- Maintain todo/debugging/queue
- Run ledger: `runs/<run_id>/spec.md`, `submit.json`, `results.md`
