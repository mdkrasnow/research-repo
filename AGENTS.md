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

### v10 CIFAR sanity status (2026-05-20)
v10_hard_example: 150ep variant harness PASS. Final FID 13.40 (5K) vs v00 R4 vanilla 14.17 (beats by 0.77).
Mining ratio L_v10/L_EqM stable 1.047-1.049 across 150ep — non-saturating signature, central differentiation
from v02 cosine saturation at EqM-B/2 IN-1K scale. Per CIFAR rule above: still requires Phase 1b/2 IN-1K
confirmation; not publishable on its own.

### Branch B-Both retired (2026-05-23)
CAFM-on-EqM port FAILED Phase 1: FID 341.25 vs vanilla 31.41. Mechanism bug, not retunable.
See `documentation/postmortem-cafm-eqm-2026-05-23.md`. Branch pivots to v10-only.
Workshop story: "first adaptive hard-negative mining for regression-target generative models" (per VeCoR §7).
CAFM-EqM port code preserved in `experiments/cafm_eqm/` for record; unused going forward.

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

### Smoke-time sample probe (mandatory for new losses on generative models)

Lesson from CAFM-EqM Phase 1b failure (postmortem 2026-05-23): "loss finite + exit 0" smoke is necessary but NOT sufficient. CAFM smoke v14 passed all loss-finiteness checks; Phase 1b then produced FID 341 (10× worse than vanilla 31.41) — visible at ckpt step 5000 already.

For any NEW loss on a generative model:
- Smoke MUST sample ≥16 images from the resulting checkpoint and either eyeball or run a tiny FID (≤2K samples) against a small ref set.
- Cost: typically <5 min on the smoke hardware. Catches mechanism bugs that loss curves hide.
- This applies to discriminator losses, adversarial losses, contrastive losses, EBM-style losses — any objective whose loss-curve shape doesn't directly predict sample quality.

### Discriminator-based loss check (specific failure modes)

Adversarial losses have a distinct failure pattern that monotonic-decrease checks miss:
- Healthy: dis loss oscillates around its baseline; gen loss oscillates; both bounded.
- Failure (dis dominates): dis loss collapses one-sided toward 0 (e.g. 2.0 → 0.03). Gen stuck with high adv loss. Gen is being pushed into an unrecoverable hole.
- Failure (gen dominates): dis loss explodes upward, gen loss collapses to 0. Mode collapse.

When evaluating an adversarial smoke: check OSCILLATION, not monotonic decrease. One-sided dis crush = STOP, mechanism bug, NOT a retune target.

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

### Positive and negative controls (mandatory)

Every experiment testing whether a mechanism works MUST ship with both controls in the same run.
A treatment number alone is uninterpretable — you cannot tell "mechanism works" from "task is
trivial" or "harness is broken" without the controls bracketing it.

- **Positive control (upper bound):** an arm GIVEN the answer / true mechanism (e.g. oracle symmetry,
  known label, ground-truth operator). Confirms the metric can move and the target is achievable in
  this setup. If the positive control fails, the harness/metric is broken — fix that before reading
  the treatment.
- **Negative control (floor / null):** an arm that SHOULD fail (e.g. vanilla baseline, random/linear
  operator where a nonlinear one is needed, shuffled labels). Confirms the metric isn't trivially
  satisfied and that any treatment gain is real, not leakage or a too-easy task.
- Read the treatment ONLY in the band between the two controls. Treatment ≈ negative → mechanism
  dead. Treatment ≈ positive → mechanism works. Treatment outside the band (above positive / below
  negative) → bug, leak, or metric artifact — debug before believing it.

Live example: latent-symmetry rung (`experiments/symmetry_toys/latent_symmetry.py`) — ORACLE
(positive: given true latent rotation) + DISC_LINEAR (negative: linear op can't fit a nonlinear
hidden symmetry) bracket DISC_NONLIN (treatment). Without ORACLE a zero DISC result could mean
"gap unfillable"; without DISC_LINEAR a nonzero result could mean "task trivial." Both needed.

This composes with the smoke-time sample probe and diagnostics rules above: controls tell you the
RESULT is real; smoke tells you the CODE is real. Run both.

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
- **Pick by ACTUAL wait time, not habit. ALWAYS check before submitting:** `squeue -u $USER --start`, `squeue --start -p <part>` (est start), `sinfo -p <part> -o '%P %a %D %t %G'` (idle nodes) across gpu_test/seas_gpu/gpu. Submit to best real priority NOW. (Lesson 2026-06-29: `gpu` backlogged ~24h overnight while seas_gpu cleared hourly + gpu_test idle — lost a night.)
- **If a job fits gpu_test, put it there** (higher priority, 12h cap). Fits = inference/smokes/≤12h. gpu_test cards are MIG-sliced A100 3g.20gb: **single-GPU works** (inference light); **4-GPU DDP does NOT** (MIG can't multi-rank NCCL → "Duplicate GPU detected"). Convert multi-GPU inference to `--nproc_per_node=1 --batch-size 16` to use gpu_test. Verified clean for B/2 selection+segmented (n=10k ≈ 0.7 img/s single-GPU). gpu_test QOS ~2 concurrent/user → overflow to seas_gpu.
- ≥ 12h runtime / multi-GPU DDP → `seas_gpu` (full A100, fast all night, dodges MIG/bad nodes). `gpu` only if both backlogged AND >12h — chronic backlog, avoid for short jobs.
- gpu_test 20G slices = OOM for mining variants (PGA = 3-4x activation memory). Mining variants → `seas_gpu`.

### QOS / partition diversification
4+ simultaneous submissions → split across `gpu_test` + `gpu` to avoid `QOSMaxSubmitJobPerUserLimit`.

### Early polling
After every submit → first poll ~60s post-submit (catch init errors: module load, pip, clone). Then resume normal interval (120s for short jobs, 1800s for long).

### Auto-pruner standing infrastructure (long-running parallel trains)
When ≥3 simultaneous train jobs write ckpts every 5K steps on home filesystem:
- Launch `slurm/jobs/prune_all_active.sbatch` on shared partition. Pruner watches all `imagenet1k_*` dirs, keeps anchors {5000, 65000} + latest 2 per dir, deletes ALL ckpts in `imagenet1k_*smoke*` dirs (smokes never need them), AND nukes rsync temp files (`.X.pt.YYYYY` patterns).
- Without pruner: 4×A100 v10 B/2 80ep produces ~75 ckpts × 2GB = 150GB; home quota is 100GB hard limit; train wedges on quota-deadlock (SLURM stdout pipe blocks).
- Auto-exits when no `eqm-1k|eqm-imagenet|prune-v10` jobs remain in queue.

### Rsync temp-file failure mode
Symptoms: persistent results dir grows large despite pruner running; hidden files `.0015000.pt.yJJYZ6` etc. visible in `find`.
Cause: rsync partial transfer interrupted (SLURM SIGTERM, preempt, timeout) leaves `.dst.XXXXXX` temp files. Default pruner glob `*.pt` misses them.
Mitigation: pruner_all_active.sbatch also runs `find $RESULTS_ROOT -name '.*.pt.*' -delete`. ANY new sbatch with rsync sync MUST tolerate these temps existing.

### gpu_requeue MIG roulette
gpu_requeue picks nodes from a pool that includes MIG (sliced) cards. MIG-sliced cards cannot do multi-rank NCCL — `Duplicate GPU detected: rank 1 and rank 0 both on CUDA device 6000`. Symptoms: job FAILED ~3min after start, NCCL `ncclRemoteError` in stderr.
Mitigation: for multi-GPU DDP jobs, use seas_gpu (full A100/H200) unless preempt-recoverable + lucky on node assignment. Single-GPU jobs OK on gpu_requeue.

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
