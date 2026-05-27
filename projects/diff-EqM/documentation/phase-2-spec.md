# Phase 2 Spec — v10 Multi-seed Welch t-test

Status: IN FLIGHT (jobs 16362498 seed 1, 16362499 seed 2). Submitted 2026-05-27 after Phase 1 PASS (FID 29.01 vs vanilla 31.41).

## Objective
Establish that v10's Phase 1 FID gain (2.40 over vanilla 31.41) is **statistically reproducible across seeds**, not a single-seed lucky run.

## Pre-registered gate (per CLAUDE.md + summer-2026-plan.md)
- **3 seeds** of v10 IN-1K-256 EqM-B/2 80ep (seed 0 done = 29.01; seed 1 + seed 2 in flight).
- **3 seeds** of vanilla EqM-B/2 80ep IN-1K-256. Currently only seed 0 trusted (31.41). Need seed 1 + seed 2 baselines too OR use established paper-level variance (~0.5-1 FID std).
- **Welch t-test** between v10 mean and vanilla mean: **p < 0.05**.
- **Mean gain >= 1.0 FID** vs vanilla.

## Decision tree
- **PASS** → proceed to Phase 3 scaling curves (S/2, L/2, XL/2). Currently in flight as smokes + S/2/L/2 vanilla baselines.
- **FAIL on Welch** (high variance, one seed regresses) → diagnose: is it ckpt/sampler stochastic noise, or true seed sensitivity? If sampler noise → run 2 more seeds. If true seed sensitivity → mining mechanism is fragile at scale → kill and pivot to v11 equivariant fallback.
- **FAIL on margin** (all 3 seeds positive but mean gain < 1.0) → marginal, not paper-worthy. Try λ retune (0.3 already in flight as 16371668).

## Vanilla seed-variance gap
We currently only have ONE seed (seed 0) of vanilla EqM-B/2 80ep IN-1K (FID 31.41). For a proper Welch t we'd need vanilla seeds 1+2 too (~300 GPU-h). Options:
1. Run vanilla seeds 1+2 in parallel (most defensible — required for paper anyway).
2. Use prior literature variance estimate (paper doesn't report B/2 80ep std; risky).
3. One-sample t-test against fixed vanilla 31.41 (weaker; reviewers will push back).

**Recommendation**: launch vanilla seeds 1+2 as soon as QOS allows. ~300 GPU-h sunk cost is small vs risk of unpublishable result from improper test.

## Required diagnostics per seed
Same as Phase 1 (CLAUDE.md mandatory):
- clean base loss `L_clean`, mining loss `L_v10`, aux/base ratio
- `||δ||` (mean + std)
- per-step wall time
- 50K-sample FID at final ckpt (gd sampler, eta=0.003, 250 NFE, cfg=1.0)

## Expected timeline
- Seed 1 + 2 train: ~30-36h each (gpu partition pending → start variable)
- Seeds 1+2 FID eval: ~2-3h each on gpu_requeue
- Phase 2 verdict: ~2-3 days from launch (2026-05-29 to 2026-05-30)

## Active jobs
| Job ID | Run | Status |
|---|---|---|
| 16362498 | v10 seed 1 train | PENDING gpu |
| 16362499 | v10 seed 2 train | PENDING gpu |

## Open questions
- Should we add vanilla seeds 1+2 (~300 GPU-h) before Welch?
- λ=0.3 ablation (16371668) running in parallel. If it lands a better FID than λ=0.1's 29.01, do we replace Phase 2 recipe mid-flight? Default NO — Phase 2 must be on the recipe Phase 1 validated.
