# Selector replication — B/2 vanilla seed1 (checkpoint-specificity test)

**Goal:** re-run the locked B/2 metacognition selector protocol on a *second* checkpoint
(B/2 vanilla **seed1**, `final.pt`) to test whether the locked result is checkpoint-specific.
Locked reference = B/2 seed0 (`stage_b_vanilla_in1k_80ep_seed0`).

Candidate selection + rationale: see `NEXT_CHECKPOINT_DISCOVERY.md`. Same harness, same arch,
locked sampler config (gd eta=0.003 / 250 / cfg=1.0) applies verbatim → zero loader risk.

## Step 1 — Separability diagnostic (job 25994331, gpu_test, COMPLETED 45m)
3000 samples, sample+log → quality labels → scores → analyze.

| metric | B/2 seed1 (this run) | locked B/2 seed0 |
|---|---|---|
| best scalar within-norm AUROC | 0.618 (s8 norm-osc) | scalars dead ~0.60–0.63 |
| label sanity (s4 latent-NN raw) | **0.577 (WEAK — flagged)** | stronger |

Scalar energies DEAD (≈0.60) — **reproduces** locked "energy scalars dead" finding.

## Step 2 — Learned probe over descent dynamics (CPU, `learned_probe.py` + `probe_validate.py`)

| probe group | B/2 seed1 within-norm AUROC | locked B/2 seed0 |
|---|---|---|
| MAG-only (norm floor) | 0.626 | ~0.61 |
| **SHAPE-only (de-confounded)** | **0.742** (learned) / **0.746±0.007** (5-seed held-out) | **0.813±0.002** |
| FULL (shape+mag) | 0.745 | — |

**Mechanism REPLICATES on 2nd checkpoint:**
- SHAPE ≫ MAG (0.746 vs 0.626) — descent *shape* carries the signal, magnitude is the floor ✓
- de-confounded within-norm ✓ and held-out across 5 CV seeds ✓
- FULL ≈ SHAPE — adding magnitude features does not help ✓ (matches locked shape-not-mag ablation)

**Quantitative gap:** absolute de-conf AUROC **0.746 < locked 0.813** (~0.07 lower) and below the
0.80 action threshold. Likely *partly* attributable to this run's weak label sanity (s4=0.577),
which depresses all training-label-based AUROCs. Whether the residual gap is checkpoint-specific
or label-pipeline noise is **resolved by the selector arms below**, whose FID readout is
label-independent.

## Step 3 — Selector arms (FID, label-independent) — IN PROGRESS
`probe_gated.sbatch`, probe artifact = `runs/b2_seed1_repl/probe_artifact.npz` (dim=30).
- Smoke: NUM_SLOTS=512 R=3, gpu_test single-GPU — **job 26013773** (de-risk wiring).
- Full (pending smoke): NUM_SLOTS=50000 R=3, seas_gpu 4-GPU — arms long250 / r3rand / r3energy / r3probe@50.
- Locked seed0 headline (5-seed CI): r3probe@50 **24.66±0.16** vs r3rand 27.95 vs long250 28.10.
- Replication question: does r3probe@50 beat r3rand by a comparable margin on seed1?

## Interim verdict
Mechanism (shape-not-magnitude, de-confounded, held-out) **replicates** on a second checkpoint —
not a seed0 artifact. Absolute probe AUROC lower (0.746 vs 0.813); selector FID arms will decide
whether the *actionable* inference-time gain transfers. **No promotion / claim change yet.**
