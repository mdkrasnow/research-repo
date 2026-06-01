# Exp1 — NFE/sampler robustness: findings

Vanilla EqM vs ANM (v10) EqM-B/2 IN-1K, frozen-checkpoint sampler sweep.
Code: `experiments/exp1_sampler_robustness/`. Status as of 2026-06-01.

## Runs
- `17828606` (gpu, 1-GPU, 5k samples): FAILED at 58/80 (SIGPIPE, see debugging.md). 57 valid cells recovered from job.log; **anm_ngd entirely lost**.
- `18003763` (gpu, 1-GPU): merge/resume — NFS dir `results/exp1_run_5k_merged` seeded with the 57 recovered cells, runs the ~23 missing cells, then analyses the full 80. RUNNING.

## Interim result (gd complete; ngd pending merge) — TENTATIVE
At **5k samples** (absolute FID inflated vs the 50k baselines — vanilla_gd nfe250 ≈ 46 here vs 31.41 @50k; read deltas, not absolutes):

| Regime | ANM vs vanilla (gd) |
|---|---|
| Converged: nfe≥100, or nfe50 + step_mult≥1.5 | **ANM wins ~2 FID** (nfe250: 46.5→44.6 @step0.5, 45.7→43.8 @step1.0) |
| Low-NFE / underconverged: nfe10, nfe25, nfe50@step≤1.0 | ANM **loses slightly** (+0.02 .. +1.3 FID); both FIDs 200–390 (useless quality) |

**ANM wins 7/18 gd cells — all in the converged regime.**

### Honest read
ANM improves **final/converged sample quality** (reconfirms the known ~2-FID v10 gap), but does **NOT** improve low-NFE sampler robustness — at low NFE it is marginally worse. Per the pre-registered criteria this is the *"better default FID but not more robust"* outcome (a partial/negative result for the robustness hypothesis), NOT the hoped-for broad low-NFE/AUC win. Final verdict needs the ngd half + full-grid AUC (job 18003763).

## Caveats (do not misread)
- **5k ≠ paper-grade.** All flagged `UNRELIABLE_small_n`. 50k is the resume follow-up (`RESUME=1`, larger wall). Absolute FIDs are sample-count inflated; only vanilla-vs-anm deltas at matched cells are meaningful.
- **step_mult couples into the time schedule** (`t += stepsize`) — axis is "step+time mult", not pure step size.
- `nfe_field = nfe − 1` (upstream loop length), reported truthfully.
- Reference is frozen + deterministically built (seed 123, no shuf), reused across all cells; merge run reproduces it identically.

## Next
1. Merge `18003763` → full 80-cell AUC / pairwise_delta / nfe-to-match. Confirm whether ngd shows the same converged-only pattern.
2. If signal holds, 50k paper-grade resume for the publishable table.
3. Pair with Exp2 (off-trajectory field robustness) for the mechanistic claim.
