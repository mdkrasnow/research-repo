# Exp1 — NFE/sampler robustness: findings

Vanilla EqM vs ANM (v10) EqM-B/2 IN-1K, frozen-checkpoint sampler sweep.
Code: `experiments/exp1_sampler_robustness/`. Status as of 2026-06-01.

## Runs
- `17828606` (gpu, 1-GPU, 5k): FAILED 58/80 (SIGPIPE, debugging.md). 57 cells recovered from job.log; anm_ngd lost.
- `18003763`/`18361440`/`18367691`: merge/resume attempts, each killed by a *different* infra failure (gpu-partition starvation; corrupt-JPEG C-segfault in ref build; home quota full → readonly). See debugging.md.
- **`18369465` (seas_gpu, 1-GPU, COMPLETED 5h49m)**: merge routed through `/n/netscratch` (home unwritable). 57 seeded cells skipped + 23 run, **0 failures → full 80 cells**. Data: `documentation/exp1_data/*.csv` (results/auc/pairwise/nfe_to_match). Plots on scratch `exp1_run_5k_merged/plots`.

## FINAL result — full 80 cells @5k samples
Absolute FID inflated vs 50k baselines (vanilla_gd nfe250 ≈ 45.7 here vs 31.41 @50k). **Read deltas, not absolutes.** Consistent across BOTH samplers:

| ANM − vanilla (− = ANM better) | gd | ngd |
|---|---|---|
| Global log-NFE FID AUC | −0.68 | −0.59 |
| Low-NFE mean FID (10/25/50) | −0.06 | −0.05 (≈tie) |
| Stress FID (low-nfe × step≥1.5) | −0.48 | −0.42 |
| Best/converged FID | **−1.91** | **−1.78** |

- **ANM wins 9/20 cells per sampler; 7/8 in the converged regime (nfe≥100), 0/8 at low-NFE (≤25).**
- NFE-to-match vanilla-250: full 250 at step≤1.0; **100 steps at step 1.5/2.0** (both samplers, 2.5× fewer).
- nan/divergence = 0 in all 80 cells; no collapse at any (nfe, step).

### Verdict (honest, vs pre-registration)
ANM's gain is a **~2-FID improvement concentrated in the converged regime (nfe≥100), robust across gd+ngd and all step sizes** — reconfirms the known v10 FID gap and extends it to the ngd sampler. The global AUC edge is **real but small (~0.6) and entirely driven by converged cells**. **ANM does NOT improve low-NFE sampler robustness** — at nfe≤25 it is statistically tied with vanilla (deltas < bootstrap noise). Pre-registered criteria 1 (lower AUC) technically PASS but trivially; criterion 2 (better low-NFE) FAIL; criterion 3 (fewer NFE-to-match) PARTIAL (only step≥1.5). **Outcome: "better final quality, not broad robustness"** — a partial/negative result for the robustness hypothesis. Honest paper framing: *ANM improves converged-regime FID consistently across samplers; it is not a sampler-robustness method.*

### Caveats / next
- 5k ≠ paper-grade. Re-run at 50k (`RESUME=1`, larger wall) before any table; only deltas meaningful here.
- The mechanistic claim (does ANM improve off-trajectory field alignment?) is NOT settled by this sampler-level result — see Exp2 (off-trajectory field robustness).

## Caveats (do not misread)
- **5k ≠ paper-grade.** All flagged `UNRELIABLE_small_n`. 50k is the resume follow-up (`RESUME=1`, larger wall). Absolute FIDs are sample-count inflated; only vanilla-vs-anm deltas at matched cells are meaningful.
- **step_mult couples into the time schedule** (`t += stepsize`) — axis is "step+time mult", not pure step size.
- `nfe_field = nfe − 1` (upstream loop length), reported truthfully.
- Reference is frozen + deterministically built (seed 123, no shuf), reused across all cells; merge run reproduces it identically.

## Next
1. Merge `18003763` → full 80-cell AUC / pairwise_delta / nfe-to-match. Confirm whether ngd shows the same converged-only pattern.
2. If signal holds, 50k paper-grade resume for the publishable table.
3. Pair with Exp2 (off-trajectory field robustness) for the mechanistic claim.
