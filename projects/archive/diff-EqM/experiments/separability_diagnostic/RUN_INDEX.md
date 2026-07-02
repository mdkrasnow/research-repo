# RUN INDEX — EqM trajectory-metacognition

Single source of truth for every run feeding the metacognition result. Branch:
`eqm-trajectory-metacognition` (off `main`). Last updated 2026-06-14.

## Checkpoint (fixed across all runs)
- **Model:** EqM-B/2, 80ep, IN-1K-256, vanilla.
- **Path:** `results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt` (2.09 GB).
- **Trusted baseline FID:** 31.41 (50k, paper 32.85).
- **Sampler:** GD, η=0.003, cfg=1.0, 250 steps, 256px, VAE-ema latent 4×32×32.

## Eval stats
- **Reference:** `in1k_reference_stats.npz` (50k real IN-1K, Inception pool3 2048-d). Trusted.
- **Proxy reference (pool-rejection only):** 10k real subset — RELATIVE ordering valid only.

---

## Runs

### R0 — separability pool (local cached, COMPLETE) ✅
- **Dir:** `runs/b2_vanilla/`
- **What:** 3000 GD samples with per-step trajectory logging (norm/dot/l2/step_dot).
- **Artifacts:** `logs/traj_rank0.npz` (101 MB, 3000×250 trajectories), `labels.csv` (3000), `scores.csv` (3000×4 regimes = 12000 rows), `probe_artifact.npz` (dim-30 SHAPE probe w,b,μ,σ), `thresholds.json`.
- **Labels:** Inception pool3 k-NN dist to 20k real IN-1K, k=3, crisp 25% tails → good=750 / garbage=750 / ambiguous=1500. Label sanity s4 raw AUROC = 0.592 (weak but usable; one retune already spent q0.40/0.30→0.25/0.25).
- **Cluster jobs that produced it:** 22507052 (full sample), 22518284 (label retune), 22539141 (dynamics scores).
- **Key results:**
  - Endpoint energy scalars DEAD: dot s1 within-norm 0.609, path-integral s3 0.605; dumb latent-NN s4 0.627 BEATS them.
  - SHAPE probe GREEN: within-norm de-confounded 0.816, held-out 30% 0.818±0.012 (5 seeds), dim 30 ≪ 1200 train.
  - Pool-rejection payoff: probe-keep beats random floor every keep-frac 0.5–0.9, recovers ~45% inception-oracle gain (PROXY FID, 10k ref).

### R1 — in-line restart sampler @ 15k (cluster, COMPLETE) ✅
- **Job:** 22619022 (1 GPU, gpu_test; multi-GPU partitions queued out).
- **What:** probe-guided best-of-R=3. Per slot: draw 3 (restart=fresh noise, same class), keep argmin-probe-P(garbage). Three arms from SAME draws.
- **Sample count:** 15000 slots × R=3.
- **Eval:** Inception features vs trusted 50k `in1k_reference_stats.npz`.
- **FID table:**

  | arm | role | n | FID |
  |---|---|---|---|
  | vanilla (draw 0) | neg control / baseline | 15000 | 29.53 |
  | **probe-gated** (argmin P_garb) | treatment | 15000 | **27.84** |
  | oracle (argmin inception-NN) | pos control / ceiling | 15000 | 17.75 |

- **Reading:** vanilla 29.53 ≈ trusted 31.41 → pipeline valid, absolute FID trustworthy. Probe Δ1.69 over vanilla, inside neg/pos band → real. Recovers 14% of oracle (restart harder lever than rejection's 45%).
- **Seeds:** seed0 only (1 sampler seed). **Consistency across seeds NOT yet established** — this is the Phase 1 gap.

### R2 — smoke (cluster, COMPLETE)
- 512 slots, R=3: probe 98.2 < vanilla 99.7 < oracle 81.6 (direction right; small-N FID inflated). Pipeline sanity only.

---

## Scripts (pipeline order)
| stage | script | runs on |
|---|---|---|
| 1 sample+log | `sample_with_logging.py` | GPU |
| 2 labels | `compute_quality_labels.py` | GPU |
| 3 scores | `compute_scores.py` | CPU/GPU |
| 4 analyze (scalars) | `analyze.py` | CPU |
| 5 learned probe | `learned_probe.py` | CPU ~1s |
| 5b held-out + ablation | `probe_validate.py` → `probe_artifact.npz` | CPU |
| 6 pool-rejection payoff | `fid_payoff.py` (+ `fid_payoff.sbatch`) | GPU |
| 7 in-line restart | `probe_gated_sample.py` + `fid_gated_agg.py` (+ `probe_gated.sbatch`) | GPU multi |
| posthoc robustness | `robustness_analysis.py` (label-sweep) | CPU |
| posthoc dynamics | `dynamics_probe.py` (learned vs baselines) | CPU |
| synthetic tests | `test_posthoc.py` (3 cases, ALL PASS) | CPU |

## Docs
- `FINDINGS.md` — full KILL→WEAK→GREEN arc + @scale.
- `POSTMORTEM.md` — energy-scalar death.
- `METACOGNITIVE_RESCUE_SPEC.md` — threshold-adaptive rescue pilot (unbuilt).

## Git / hygiene state
- Branch `eqm-trajectory-metacognition` carries all sep-diag work.
- Unpushed local commits (main 2 ahead of origin): **d71d4dc** (posthoc ladder + dynamics probe + gated sampler), **73c0423** (@scale FID 27.84). Hold per "do not push until upstream ready".
- **Tangle note:** commit **0a02e96** (rc-hpm, already pushed) folded 3 sep-diag edits (`analyze.py`, `compute_quality_labels.py`, `sample_with_logging.py`) via an `--amend`. Already in published history → NOT rewriting (risk > benefit; edits are minor and functional). Documented here instead.

## Open / cluster-gated
- 50k consistency, multi-seed (Phase 1) — needs cluster GPU + SSH.
- Online equal-NFE adaptive sampler at scale (Phase 2) — local synthetic smoke OK, scale cluster-gated.
- Maze planning prototype (Phase 3D) — CPU-local, autonomous.
