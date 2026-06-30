# Claude Code to Pi Memory Migration Report

## Summary

- Repo: `research-repo` (`/Users/mkrasnow/Desktop/research-repo`)
- Date: 2026-06-28
- Pi version: 0.73.0
- Pi memory dir: `~/.pi/agent/memory`
- Claude memory source dir: `~/.claude/projects/-Users-mkrasnow-Desktop-research-repo/memory/` (unambiguous path match; 18 .md files)
- pi-memory package: installed via `npm:@zhafron/pi-memory` (git install failed — husky/npm code 127; npm fallback succeeded)
- Files changed:
  - `CLAUDE.md` (reduced to `@AGENTS.md` import + Claude-only stub; old content preserved in AGENTS.md)
  - `~/.pi/agent/settings.json` (merged `pi-memory` config + `skills` path; packages list got pi-memory)
  - `~/.pi/agent/memory/{MEMORY,USER,IDENTITY}.md` (created/written)
  - `~/.pi/agent/memory/claude-import/` (19 files copied verbatim)
  - Backups in `.pi/memory-migration-backups/20260628-214600/`
- AGENTS.md: PRESERVED unchanged (345 lines) — it was already the canonical always-on project ruleset (created earlier this session as exact copy of CLAUDE.md).

## Imported to AGENTS.md
- No change. Existing AGENTS.md already contains the full always-on repo instructions
  (scope, gating discipline, SLURM/cluster rules, job-tracking protocol, variant proposal
  template, research process rules). Per the model, existing AGENTS.md is preserved.

## Imported to Pi MEMORY.md
- Repo scope (single active project diff-EqM; others dormant), north-star deadlines, phase-gating rules.
- Key results: vanilla baseline FID 31.41 (trusted); v10 IN-1K 3-seed win 27.58±0.36; v10/v02 CIFAR; B-Both retired (FID 341); Exp2/Exp3 outcomes.
- Active/reopened: EqM separability descent-shape probe 0.813; symmetry v17 MorphismGym positive + v16 distrust; RC-HPM.
- Debugging lessons: harness FID gap, smoke must sample images, adversarial oscillation check, dataloader workers, exhaustive config extraction.
- Cluster/infra: holylabs vs home quota deadlock, MIG/NCCL, OOM partitions, QOS split, auto-pruner/rsync temps, ImageNet blocker, FID never auto-approved.
- Job tracking: pipeline.json active_runs source of truth; FID sbatch needs explicit GIT_SHA.
- Pointers to all 19 verbatim files in `claude-import/`.

## Imported to USER.md
- Autonomous iteration (don't ask permission for routine continuation).
- Append-only to results.tsv, never overwrite (incl. SKIP/CRASH rows).
- Concise reports.
- Commit attribution: sole credit to mdkrasnow; no Co-Authored-By / generated-with trailers.

## Imported to IDENTITY.md
- Mechanism-first research engineer; controls (positive+negative) mandatory; read treatment in-band.
- Parallel/batched independent work.
- Simplicity-first (3+ repeats before frameworks).
- Surface state-file conflicts before acting; pre-registered gate beats judgment.

## Pi Skills Created
- None fabricated. Pi configured to reuse Claude skills via `skills: ["~/.claude/skills"]`
  in settings. Repo workflows already live as `.claude/commands/` slash commands + skills
  (autoresearch, dispatch, check-results, etc.); not duplicated into `.pi/skills/` to avoid drift.

## Sources Inspected
- `./CLAUDE.md`, `./AGENTS.md` (repo)
- `~/.claude/CLAUDE.md` (global: model default, commit attribution, parallel-exec, meta-reasoning)
- `.claude/settings.json`, `.claude/settings.local.json`, `~/.claude/settings.json`, `~/.claude/settings.local.json` (no `autoMemoryDirectory` set)
- `~/.claude/projects/-Users-mkrasnow-Desktop-research-repo/memory/*.md` (18 files — all imported)
- No `.claude/rules/` or `~/.claude/rules/` present.
- 6 other Claude project memory dirs exist but belong to other repos — not imported.

## Skipped / Stale / Duplicate Items
- Original `MEMORY.md` had stale/over-length task state (q211/q220/q225 IRED job IDs, algebra-ebm
  phase notes) for DORMANT projects — NOT promoted into curated Pi MEMORY.md (full text still in
  `claude-import/MEMORY.md` if ever needed). Reason: dormant per repo scope; old task state low value.
- Global ~/.claude/CLAUDE.md parallel-exec/meta-reasoning prose — condensed into IDENTITY.md rather than copied verbatim. Reason: dedupe, stays global anyway.

## Conflicts
- `~/.claude/CLAUDE.md` says "Use Claude Opus 4.6 as default" — Claude-specific, irrelevant to Pi
  (Pi default model = qwen36 via dgx-spark). Left as-is in each tool's own config; not migrated. No action needed.
- None blocking.

## Validation Results
- AGENTS.md exists (345 lines); CLAUDE.md exists (10 lines, first line `@AGENTS.md`).
- Pi MEMORY.md (91) / USER.md (18) / IDENTITY.md (15) exist.
- claude-import/: 19 files.
- `python3 -m json.tool ~/.pi/agent/settings.json` → valid.
- Pi smoke test (`pi -p ...`) PASSED: Pi listed MEMORY/USER/IDENTITY and recalled the v10 IN-1K FID 27.58 win.

## Suggested Next Steps
- Start Pi in this repo.
- Ask Pi: "Summarize what you remember about this repo from pi-memory."
- Verify it surfaces diff-EqM scope, v10 win, and cluster/holylabs rules.
- Optional: trim AGENTS.md toward the <200-line target by moving the longest historical
  prose (postmortems, variant logs) into Pi memory — deferred to avoid losing load-bearing rules.
