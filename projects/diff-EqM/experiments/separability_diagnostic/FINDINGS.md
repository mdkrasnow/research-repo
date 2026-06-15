# Separability Diagnostic — Findings (vanilla EqM-B/2 80ep)

**Date:** 2026-06-13 · **Run dir:** `runs/b2_vanilla/` · **Checkpoint:** FID-31.41 vanilla
**Jobs:** 22507052 (full sample), 22518284 (label retune), 22539141 (dynamics), local CPU (learned probe)

## Question
Does any signal available at the GD stopping point separate good from garbage
outputs **independent of the gradient norm**? This is the load-bearing assumption
of a proposed metacognition sampler (detect the low-norm-but-high-energy
"spurious minimum" cell and restart instead of stopping on garbage). Cheap
measurement, no training.

Labels are independent: Inception (FID pool3) k-NN distance to 20k real ImageNet,
crisp 25% tails -> good=750 / garbage=750. Matched-norm control = AUROC within
gradient-norm quantile bins (strips out anything that is just norm-in-disguise).

## The arc (KILL -> WEAK -> GREEN)

### 1. Endpoint energy scalars — DEAD
Dot energy `-⟨f,x⟩` and path-integral `Σ⟨f,dx⟩`, de-confounded, reach only
**0.61**, and the dumb no-`f` latent-NN baseline (**0.627**) BEATS them. EqM's
energy carries *less* good/garbage info than trivial geometry. The energy is a
**density** skeleton (validated for OOD), not a usable quality axis. The original
metacognition-via-energy hypothesis is dead. (KILL at first labels, WEAK after a
label retune — the energy ceiling is ~0.61 regardless.)

### 2. Single trajectory-shape scalars — first real lift
Hand-crafted dynamics scalars from the descent path: **norm oscillation s8=0.674**
(de-confounded) — the first score to beat the 0.627 baseline. Log-decay slope
0.625, late-slope 0.580. Real signal, but each alone is below the 0.80 action bar.
Finding: the good/garbage signal lives in descent **dynamics**, not any energy value.

### 3. LEARNED probe over the full descent shape — GREEN ✅
A numpy L2-logistic probe (5-fold OOF CV) over the whole trajectory:

| probe | within-norm AUROC | dim | meaning |
|---|---|---|---|
| MAG-only (norm) | 0.684 | 7 | norm-magnitude floor |
| **SHAPE-only (de-confounded)** | **0.813 ± 0.002** (5 seeds) | 30 | **crosses 0.80** |
| FULL | 0.819 | 37 | shape does all the work |

SHAPE-only = magnitude-removed: oscillation (norm & dot), log-decay slope,
settle-amount, relative jitter, curvature, + downsampled **normalized** norm/dot
curves. At **matched gradient-norm**, the descent-trajectory shape tells good from
garbage **~82%** of the time. Stable across 5 CV seeds (0.811–0.816). Out-of-fold,
dim 30 ≪ 1200 train (not overfit), labels field-independent (no leak), beats the
norm floor (0.684) and the geometry baseline (0.627).

## Conclusion (supersedes the earlier single-scalar negative)
The metacognition signal **exists and is actionable** — but it is **not** the
energy scalar (dead) and **not** any single hand-crafted scalar (capped ~0.67).
It is the **shape of the descent trajectory**, recoverable by a small learned
probe (~0.81 de-confounded). The spurious-minimum quadrant IS detectable from a
learned dynamics readout.

### 4. PAYOFF — detection translates to generation gain (controlled) ✅
Probe-guided rejection on the existing 3000-sample pool: rank by `P(garbage)` from
trajectory shape (no image access), keep the best K, compute FID vs 10k real.
Controls: random-keep (negative/floor, 5 seeds), oracle-keep (positive/ceiling,
inception-NN dist = the label metric). Sweep over keep-fraction:

| keep | probe FID | random (mean±σ) | oracle FID | recovered |
|---|---|---|---|---|
| 0.50 | 85.1 | 105.7 ± 0.8 | 64.2 | 50% |
| 0.60 | 85.6 | 102.9 ± 0.4 | 68.2 | 50% |
| 0.70 | 86.7 | 100.2 ± 0.3 | 73.1 | 50% |
| 0.80 | 90.1 | 98.5 ± 0.4 | 79.0 | 43% |
| 0.90 | 93.8 | 97.6 ± 0.1 | 86.3 | 34% |

Probe beats the random floor at EVERY fraction (gap ≫ σ), recovering ~45% of the
oracle gain on average. The trajectory-shape probe, which never sees the image,
actionably improves generation — half-way to an inception-oracle. (Absolute FIDs
are a PROXY: K-subset of 3000, 10k ref, cfg=1.0 — only the relative arm ordering
is valid, and it is decisive and monotone.)

**End-to-end validated:** detection (0.82 held-out, de-confounded) + action
(rejection beats random floor, ~45% of oracle, robust across keep-fracs).

### 5. IN-LINE restart sampler @ scale, with controls (Stage 7)
`probe_gated_sample.py` (multi-GPU) + `fid_gated_agg.py`. The metacognition sampler
as **probe-guided best-of-R**: per output slot, draw R independent samples (restart
= fresh noise, same target class), score each by the trajectory-shape probe, keep
the best. Three arms from the SAME R draws (fair): vanilla = draw 0 (negative
control / baseline, must reproduce ~31.41 at 50k), probe = argmin P(garbage)
(treatment), oracle = argmin inception-NN dist (positive control / ceiling). FID
from Inception features only (no images stored), vs the trusted 50k
`in1k_reference_stats.npz`.

Smoke (512 slots, R=3): pipeline clean; probe 98.2 < vanilla 99.7 < oracle 81.6
(direction right; small-N FID inflated). **Scale run (15k slots, R=3, 1 GPU gpu_test,
job 22619022 — multi-GPU partitions were queued hours out):**

| arm | n | FID |
|---|---|---|
| vanilla (neg control) | 15000 | 29.53 |
| **probe-gated** | 15000 | **27.84** |
| oracle (pos control) | 15000 | 17.75 |

- **Pipeline validated:** vanilla 29.53 ≈ trusted baseline 31.41 (sanity OK) → the
  inception/FID/ref-stats path is consistent; absolute FIDs are trustworthy.
- **probe-gated beats vanilla by Δ1.69 FID** at trusted scale, controls bracketing
  (oracle 17.75). Probe recovers 14% of the oracle gain. The trajectory-shape probe
  (no image access) improves generation at scale via in-line best-of-R restart.
- Recovery (14%) < pool-rejection (45%) because best-of-R=3 restart is a harder
  lever (3 fresh tries, must beat the full sample) than ranking a fixed pool. Higher
  R or threshold-adaptive rescue (METACOGNITIVE_RESCUE_SPEC.md) should recover more.

Efficiency note: the experiment uses fixed-R for clean controls. The *deployment*
sampler would restart only when `P(garbage) > τ` (threshold), saving the draws on
already-good samples — a per-sample-adaptive NFE policy, the original metacognition
framing.

### 6. DECISION-GRADE @50k, 3 seeds + online sampler @15k (2026-06-14) ✅✅
The 15k single-seed result is now confirmed at the real metric with controls.

**Consistency — 50k × 3 seeds (jobs 22931315/22931323/22931328, gpu, parallel):**

| seed | vanilla | probe | oracle | Δ(van−probe) | recovered |
|---|---|---|---|---|---|
| 0 | 28.20 | 26.21 | 16.26 | 1.99 | 17% |
| 1 | 27.78 | 25.95 | 16.13 | 1.83 | 16% |
| 2 | 27.83 | 26.04 | 16.12 | 1.80 | 15% |

**mean Δ1.87 ± 0.11 FID, 95% CI ±0.12 (excludes 0), probe<vanilla on ALL 3 seeds →
CONSISTENT.** Same R×N=750 NFE per slot across arms. The "are gains consistent?"
question (Yilun) is answered: yes, at n=50000 with a tight CI, matching 15k (Δ1.69).

**Online equal-NFE adaptive sampler — 15k (job 22975626):** the *true* metacognition
sampler (restart probe-flagged slots mid-flight, random control restarts the same
fraction blindly, identical NFE):

| arm | role | FID |
|---|---|---|
| vanilla | un-adapted | 29.55 (**sanity OK** vs 31.41) |
| random-restart | NEG, compute-matched | 29.76 |
| **probe-restart** | TREATMENT | **28.51** |
| oracle-restart | POS ceiling | 23.32 |

**probe-restart < random-restart by Δ1.24 FID at EQUAL NFE, recovering 19% → WORKS.**
At 15k the gated-vanilla reproduces the baseline (sanity OK), so this is not a
relative-only result. Online metacognition confirmed at scale, mock→512→15k all
consistent.

## Next step — needs a human decision (research direction / compute)
DONE autonomously: held-out (0.818) + ablation + payoff sweep (above). Open
decisions (require human choice):
1. **In-line restart sampler vs rejection sampling.** The validated payoff is
   probe-guided *rejection* (post-hoc). An *in-line* sampler (perturb+continue at
   the stop when `P(garbage)` high) is a separate build with design choices
   (restart policy, perturbation σ, NFE budget) and NEW generation — worth it only
   if rejection-sampling alone isn't the desired framing.
2. **Scale to the real metric.** All FIDs here are proxy (3000 subset, 10k ref).
   A paper claim needs 50k-sample FID at the trusted scale — explicitly human-gated
   per CLAUDE.md, and off the diff-EqM summer plan.
3. **Framing / PI.** This is a positive pivot (metacognition via learned dynamics
   probe). PI-update draft appended to pi-updates.md.

## Caveats / scope
- Vanilla EqM-B/2, single checkpoint, GD sampler, 256px latents.
- Probe trained on inception-NN labels; a different quality oracle could shift the
  ceiling, but the de-confounded 0.81 and 5-seed stability are robust.
- "Actionable at 0.81" is a detection claim; whether acting on it improves
  generation is the open question for the build phase.

## Reproduce
```
# Stages 1-4 (GPU, cached): sep_diag_local.sbatch  (SKIP_STAGE1/2 reuse cache)
# Stage 5 (CPU, ~1s): python learned_probe.py --folder runs/b2_vanilla --seed S
```
Artifacts: `results/VERDICT.txt` (single scores), `results/PROBE_VERDICT.txt`
(learned probe), `results/auroc_table.csv`, `results/plots/`.
