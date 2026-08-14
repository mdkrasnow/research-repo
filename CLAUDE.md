@AGENTS.md
@MEMORY.md

## Claude Code Only

<!--
All shared, always-on repo instructions now live in AGENTS.md (imported above).
Claude-specific global behavior (model default, commit attribution, parallel subagents,
meta-reasoning) lives in ~/.claude/CLAUDE.md and applies automatically.
Add only Claude-Code-specific repo notes below.
-->

## Repo memory (read/write every session)

`MEMORY.md` (repo root, imported above) is this repo's curated, git-tracked memory —
covers all projects, active and archived, not just the current single-project scope.
It is separate from Claude Code's own private per-session auto-memory.

- At the start of any session, and before any non-trivial action, check `MEMORY.md`
  for a relevant one-line pointer, and follow it into `memory/topics/<slug>.md` if so.
- If a past conversation or result is relevant to the current task, check
  `memory/YYYY-MM-DD.md` daily logs (grep by date or keyword) before assuming it isn't
  recorded anywhere.
- Write to today's `memory/YYYY-MM-DD.md` as you go during a session — don't wait
  until the end. One line per notable event: what was tried, what ran, what broke,
  what a conversation concluded.
- When something in a daily log turns out durable (matters beyond that day), promote
  it to `memory/topics/<slug>.md` (with `name`/`description`/`status` front matter) and
  add a one-line pointer to `MEMORY.md`. Keep `MEMORY.md` itself short — it's an index,
  not a place to accumulate detail.
- If `MEMORY.md` conflicts with what you observe in the current code/state, trust the
  current state and update or remove the stale entry.
