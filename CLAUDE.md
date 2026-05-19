# Claude Code Rules — diff-EqM Project (single-project focus)

This repo is now scoped to **one active project**: `projects/diff-EqM/`. All other projects are dormant. Do not work on, modify, or even read other projects unless the user explicitly redirects.

**North star**: get to a publishable result at NeurIPS 2026 workshop (deadline 2026-08-29) and ICLR 2027 main (~2026-10-01). Plan locked in `projects/diff-EqM/documentation/summer-2026-plan.md`. Read it before any non-trivial action.

This is an **agentic research project** — claude executes Phase 0 → Phase 5 with minimal supervision. Decisions follow the pre-registered gates in the summer plan. When in doubt: pre-registered gate wins, not in-the-moment judgment.

---

## Alignment-first protocol (READ BEFORE ACTING)

Before any non-trivial action (submission, config change, code change, decision about what to try next), read in order:

1. `projects/diff-EqM/.state/pipeline.json` — phase, `active_runs`, `next_action`, exit gates.
2. `projects/diff-EqM/documentation/summer-2026-plan.md` — north-star phased plan, gates, deadlines.
3. `projects/diff-EqM/documentation/phase-0-spec.md` (or the current active phase spec) — exact task breakdown.
4. `projects/diff-EqM/documentation/literature/SYNTHESIS.md` (if present) — current positioning + design implications from lit review. SUPERSEDES `related-work-differentiation.md` once written.
5. `projects/diff-EqM/documentation/related-work-differentiation.md` — initial positioning vs DAT (Wu 2025) and AEBM-Diff (Geng 2024).
6. `projects/diff-EqM/documentation/publishability-plan.md` — older strategic doc (summer plan wins on conflict).
7. `projects/diff-EqM/documentation/queue.md` — top-of-queue actions.
8. `projects/diff-EqM/results_variants.tsv` — what has already been tried.

Every proposed action must answer: **"does this move us closer to the workshop (Aug 29) or ICLR (Oct 1) submission per the summer plan?"** If no → de-prioritize or escalate.

Proxy-scale (CIFAR) gains are **not publishable** — they are filters for paper-scale (IN-1K) confirmation runs. Never scale up a result that has not passed its stage exit gate.

If state files disagree with what the user asks for, surface the conflict before acting. If a stage exit gate has not passed, say so before launching the next stage's compute.

---

## Project scope (HARD)

- Only modify files inside `projects/diff-EqM/...`, `scripts/cluster/...`, `scripts/ralph/...`, and this CLAUDE.md.
- Do NOT read, modify, or reference other projects (`projects/ired/`, `projects/algebra-ebm/`, `projects/archive/`). They are dormant.
- Do NOT spawn parallel work on other projects. Single-project focus.

---

## Job tracking protocol (mandatory)

`pipeline.json:active_runs` is the **single source of truth for every submitted SLURM job**. Nothing runs on the cluster without an entry.

**Before any action (submission, analysis, status report, planning):**
1. Read `active_runs` in `pipeline.json`. Cross-check against `squeue -u $USER` via `scripts/cluster/ssh.sh`.
2. If they disagree → reconcile first: stale entries → move to `completed_runs` with status + final metric (fetch via `sacct`); untracked running jobs → add entry with all required fields.
3. Only then plan or submit new work.

**Required fields per entry:**
```json
{
  "run_id": "<human tag, e.g. v10_in1k_seed0>",
  "job_id": "<SLURM id, including _N for array tasks>",
  "partition": "<gpu_test | gpu | seas_gpu>",
  "status": "pending | running | completed | failed | cancelled | timeout",
  "description": "<one line: what + why + phase>",
  "submitted_at": "<ISO date>",
  "git_sha": "<commit at submission>",
  "sbatch_path": "projects/diff-EqM/slurm/jobs/...",
  "expected_runtime": "<rough hours>",
  "phase": "0 | 1 | 2 | 3 | 4 | 5",
  "gate": "<which gate this run informs>"
}
```
Add `final_metric` / `exit_code` on completion; add `error` + `debugging.md` link on failure.

**On every submission** — same commit:
1. Add `active_runs` entry with all fields.
2. Commit `pipeline.json` with message naming job_id + purpose.
3. Push.

**On every completion / failure** — same session:
1. Move entry to `completed_runs`. Add outcome + final metric + duration.
2. Update `results_variants.tsv` if a metric was produced.
3. Trigger PI update per `pi-updates.md` if applicable.

**When reporting status**: always lead with reconciled `active_runs` table from file + `squeue`. Never answer "what's running?" from memory.

---

## Research process rules for EqM / ANM (project-specific, load-bearing)

### Core principle
Do not implement a new variant just because the previous one failed. Every new experiment passes a mechanism check **before code is written** — see Variant Proposal Template below.

### Baseline-first
- Stage B IN-1K-256 EqM-B/2 80ep vanilla baseline = **FID 31.41 (TRUSTED)**, ckpt `stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt`.
- For new scales (S/2, L/2) **re-baseline before comparing variants**. Treat all auxiliary-loss experiments as invalid until baseline at that scale is trusted.

### EqM objective alignment
EqM trains `f(x_γ) ≈ target(x, ε, γ)` with `target = (ε - x) · c(γ)`, c(γ) hardcoded in `transport.py:122-126`. c(γ) → 0 as γ → 1 (data manifold).

**High-risk losses (require written compatibility argument):**
- Cosine separation objectives (v02 saturated on EqM-B/2)
- Hinge on velocity norm (v01 fights c(γ) decay)
- Jacobian penalties (v09 flattens field)
- EBM-style energy losses not tied to EqM target

**Low-risk losses (preferred):**
- Auxiliary loss = EqM base loss on a perturbed/mined input (v10 hard-example)
- Losses provably bounded by base loss

### Variant proposal template (mandatory before code)

```
Variant name:
Hypothesis:
Failure mode addressed:
EqM compatibility argument:
Loss definition:
Expected diagnostics if working:
Expected diagnostics if failing:
Minimal test:
Promotion rule:
Kill rule:
```

Live example: `projects/diff-EqM/documentation/v10_hard_example_eqm_proposal.md`.

### CIFAR sanity rule
CIFAR can answer: does code run? does model collapse? are diagnostics finite? Is loss obviously broken?
CIFAR cannot answer: will this transfer to EqM-B/2? IN-1K? Is this better than vanilla EqM at scale?
Note: variant harness ≠ legacy harness — never compare FIDs across harnesses (4.7 FID gap documented).

### Diagnostics required every auxiliary-loss run
Log at min every N=200 steps:
- clean base loss `L_clean`
- auxiliary loss `L_aux` (or `L_hard` for v10)
- aux/base ratio
- field norm `||f(x_γ)||` at clean point
- field norm `||f(x_γ + δ)||` at perturbed/mined point
- perturbation norm `||δ||` (mean + std)
- mined loss before and after PGA (if mining)
- per-step wall time vs vanilla baseline
- FID or proxy eval when available

If aux/base ratio dominates → stop, retune. If aux saturates near zero → stop, diagnose.

### No literature laundering
Citation ≠ mechanism. State exactly what the paper supports and what it doesn't.

### Literature review protocol (active through paper submission)
- Per-paper notes live in `projects/diff-EqM/documentation/literature/<paper>.md` using the template in `literature/README.md`.
- Synthesis lives in `literature/SYNTHESIS.md`. Update it whenever a new HIGH-threat paper appears or 5+ new notes accumulate.
- Weekly arxiv sweep in `literature/arxiv-weekly-sweep.md`. Run Mondays. HIGH-threat hits → upgrade to per-paper note.
- Any framing change driven by lit review → propagate edits explicitly to: this CLAUDE.md, `summer-2026-plan.md`, `related-work-differentiation.md`, the current phase spec. One commit per artifact.
- Before writing new code for a variant, re-read SYNTHESIS.md §3 (mechanism design adjustments). The variant proposal template MUST cite specific notes that informed its design.

### Stop conditions (kill a direction)
- Loss contradicts EqM target geometry
- Baseline not verified
- Diagnostic signal saturated or zero
- Improvement requires post-hoc reinterpretation
- Experiment only adds complexity without testing a clear mechanism
- Same failure repeats across two reasonable HP settings
When stopped → write short postmortem before proposing next variant.

### Research loop (per experiment)
1. Baseline check
2. Mechanism note
3. Compatibility check
4. Minimal implementation + diagnostics first
5. Smoke test (collapse, saturation, overhead)
6. Decision: promote, retune once, kill
7. Postmortem

**Prove the experiment deserves to exist before writing code.**

---

## Gating discipline (the agentic guardrail)

This project runs phase-gated. Each phase has a **pre-registered exit gate**. Crossing a gate without explicit pass = invalid result.

| Phase | Exit gate | Action on fail |
|---|---|---|
| 0 | v10 CIFAR sanity no-collapse + L_hard>L_clean | redesign mining or kill v10, propose v11 |
| 1 | v10 IN-1K seed 0 FID ≤ 30.41 | 1 λ retune ∈ {0.03, 0.3, 1.0} then kill |
| 2 | 3-seed Welch t p<0.05, mean ≥1 FID gain | kill v10, paper claim collapses |
| 3 | Scaling curve monotone improvement | restrict scaling claim |
| 4 | Flow-matching transfer ≥0.5 FID | restrict claim to EqM only |
| 5 | Workshop draft ready by Aug 22 | extend timeline / fall back to ICLR only |

Maximum **1 retune per failing direction** (CLAUDE.md hard rule). No retune indefinitely loops.

---

## Commit + push discipline
- Every metric-producing result is in a commit message: "v10 IN-1K seed 0 FID 29.83 (vs vanilla 31.41) — PASS Phase 1 gate".
- Every job submission has its `active_runs` entry in the same commit.
- Every job cancellation: commit notes last observed metric + reason.
- NEVER add Co-Authored-By or Claude footer (per ~/.claude/CLAUDE.md).
- After every commit affecting cluster state → push (cluster pulls fresh per job).

---

## SLURM discipline (cluster = remote-only, GPU-only)

- SLURM commands are NEVER local. Use `scripts/cluster/ssh.sh`, `scripts/cluster/status.sh`, `scripts/cluster/remote_submit.sh`, `scripts/cluster/remote_fetch.sh`.
- Local execution = CPU only. Cluster execution = GPU (A100s).
- Cluster jobs auto-clone repo to `/tmp/project-job-$SLURM_JOB_ID`, checkout `GIT_SHA`, run, cleanup. NO manual code maintenance on cluster.
- Only `slurm/` synced to cluster (sbatch + log structure).

### Partition selection
- < 24h runtime → `gpu_test` (higher priority, 24h cap)
- ≥ 24h runtime → `gpu` (lower priority, longer time)
- For IN-1K 80ep EqM-B/2 (vanilla ~24h, v10 ≥30h) → `gpu` or `seas_gpu` per cluster guidance
- gpu_test 20G cards = OOM for mining variants (PGA = 3-4x activation memory). Mining variants → `gpu` only.

### QOS / partition diversification
4+ simultaneous submissions → split across `gpu_test` + `gpu` to avoid `QOSMaxSubmitJobPerUserLimit`.

### Early polling
After every submit → first poll ~60s post-submit (catch init errors: module load, pip, clone). Then resume normal interval (120s for short jobs, 1800s for long).

### Job verification (do FIRST every session)
1. For every `active_runs` entry: cross-check `scripts/cluster/status.sh <job_id>` or `sacct`.
2. If status mismatch → fetch logs (`scripts/cluster/remote_fetch.sh diff-EqM`), determine failure, write to `debugging.md`, move to `completed_runs`, set `phase=DEBUG` if blocking.
3. Only after verification → plan new work.

---

## Required structured outputs (after each step)

Update as applicable:
- `projects/diff-EqM/.state/pipeline.json` (always, on state change)
- `projects/diff-EqM/documentation/phase-X-spec.md` (mark tasks done)
- `projects/diff-EqM/documentation/debugging.md` (failures, root causes)
- `projects/diff-EqM/documentation/queue.md` (next actions)
- `projects/diff-EqM/results_variants.tsv` (any metric-producing run)
- `projects/diff-EqM/runs/<run_id>/...` (artifacts)
- `projects/diff-EqM/documentation/arxiv-weekly-sweep.md` (weekly scoop check)

---

## PI update protocol

Triggers (per `projects/diff-EqM/documentation/pi-updates.md`):
- stage exit gate pass/fail
- result at paper-comparable scale (any IN-1K run)
- gain confirmed across 3 seeds
- blocker needing PI input
- significant pivot (e.g., kill v10 → v11)
- scoop or external signal
- bug invalidating prior reported result

On trigger:
1. Append draft to `pi-updates.md`.
2. Set `pipeline.json:needs_user_input.value=true` with prompt pointing to draft.
3. Do NOT send. User reviews + sends.

Weekly digest day = Monday. Draft even with no trigger.

---

## Stopping

The repo's Stop hook may block stopping while work is actionable. If user input is required, set `needs_user_input.value=true` with prompt. Otherwise: complete the current sub-task, update state files, then stop.

---

## Cadence (agentic loop)

Default agentic cadence for this project:
1. Read pipeline.json, summer plan, current phase spec.
2. Identify next sub-task per current phase spec.
3. If sub-task is "submit job" → check job tracking + active_runs reconciliation first.
4. Execute one sub-task to completion.
5. Update state files (pipeline.json, phase spec, results_variants.tsv as applicable).
6. Commit + push.
7. If gate-evaluation step reached → evaluate gate explicitly against pre-registered threshold. Pass → next phase. Fail → kill or 1 retune.
8. PI update trigger check.
9. Stop (or continue to next sub-task if cheap).

No autoresearch ratchet loop on this project — the summer plan + phase specs are the loop.
