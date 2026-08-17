# MEMORY.md — research-repo curated memory

Loaded at the start of every session on this repo. Keep this file SHORT — durable
facts, standing decisions, and one-line pointers only. Details live under `memory/`.

Structure (OpenClaw-style, adapted for this repo):
- `MEMORY.md` (this file) — curated, durable, always-loaded. Facts/decisions that
  stay true across many sessions. One-line index entries link to `memory/topics/*.md`.
- `memory/YYYY-MM-DD.md` — append-only daily log. Raw session notes: what was tried,
  what ran, what broke, what a conversation concluded. Write to today's file as you go;
  don't wait until end of session.
- `memory/topics/<slug>.md` — curated write-up for one topic/finding/decision, promoted
  out of a daily log once it's durable enough to matter beyond that day. Front matter:
  `name`, `description`, `status` (active|paused|dormant|superseded).
- `memory/imports/` — memory pulled in from other tools/sessions, kept separate from
  native entries so provenance stays clear.

Promotion rule: if something in a daily log turns out to matter next week, promote a
short write-up to `memory/topics/` and add a one-line pointer here. If it stops being
true, edit or delete the entry — don't leave stale claims in this file.

Note: this is a repo-local, git-tracked memory (all projects, active + archived). It is
separate from Claude Code's own per-session auto-memory at
`~/.claude/projects/-Users-mkrasnow-Desktop-research-repo/memory/`, which is private to
this machine/account and not git-tracked. Check both; this file is the one other
collaborators (or a fresh clone) can see.

---

## Repo state (as of 2026-07-21)

- **Active project**: `projects/masked-EqM/` (fresh clone of raywang4/EqM, started 2026-07-02).
  No summer plan / phase gates exist yet for masked-EqM — see `AGENTS.md` for the hard
  single-project scope rule and alignment-first protocol.
- **Archived/dormant**: `projects/archive/diff-EqM/` (formerly active; retired 2026-07-02,
  extensive gated history — v10 hard-example mining passed Phase 1/2 IN-1K gates, see
  `memory/topics/diff-eqm-v10-history.md`), `projects/archive/algebra-ebm/`, `ired`,
  symmetry-discovery work. Do not act on these unless explicitly redirected.
- **Memory system**: this file + `memory/` set up 2026-07-21 to give a git-tracked,
  cross-session record of conversations/results spanning all projects (not just the
  active one), separate from Claude Code's private per-session memory. See
  `memory/2026-07-21.md`.
- **WFB-EqM / FBGN Stage 3** (fb_direct scalar-energy optimization track, CLOSED
  2026-08-17 as a negative result -- Stage 3B KILL fired; closure write-up at
  `projects/masked-EqM/documentation/fbgn-closure-2026-08-17.md`): certified CG Gauss-Newton reduces every one of its own minibatches yet
  worsens the held-out field objective — and so does the raw-direct negative control.
  Includes two durable corrections (the "lambda_max growth caused it" claim is retracted;
  the reduction ratio is confounded with step length) and load-bearing gotchas (probe slice
  moves with `--max-steps`; the FP64 toy is linear at default init).
  See `memory/topics/wfb-eqm-fbgn-stage3.md`.
- **Corrected-BTM / FD-scalar campaign** (branch `btm-fd-scalar`, active 2026-08-13):
  the current primary optimization thread, replacing FBGN. Tests whether the mixed
  input-parameter derivative -- not the nonexistence of a scalar transport potential --
  is what breaks explicit scalar EqM late in training, by training the corrected BTM
  solution with scalar function evaluations only. Toy gate PASSED 6/6 in two geometries;
  FD numerics validated at real B/2 scale; FD estimator VARIANCE identified as the risk.
  Includes the load-bearing label-dropout gotcha that silently destroys any finite
  difference on this architecture. See `memory/topics/btm-fd-scalar-campaign.md`.
- **Telemetry event log** (`projects/masked-EqM/telemetry/`, built 2026-08-14): the run
  instrumentation rewrite. Content-addressed `run_uid` (logical experiment) + `exec_id`
  (physical execution, since a SLURM requeue reuses the job id), a guaranteed terminal
  record via a five-layer ladder, and completeness-gated aggregation with absolute analysis
  windows. Replaces identity-by-path-parsing and run-relative windows, both of which could
  silently invert the Phase II-A conclusion. Includes declared invariants aimed at the
  `frozen_label_dropout` bug class. See `memory/topics/telemetry-event-log.md`.
- **Scalar-energy EqM proposal** (external ChatGPT conversation, 2026-07-21, unevaluated):
  candidate masked-EqM direction — train scalar `E(x)` directly instead of the vector
  field, sample by descending `∇E(x)`. Not yet run through the project's mandatory
  compatibility-check process. See `memory/topics/scalar-energy-eqm-proposal.md`.
