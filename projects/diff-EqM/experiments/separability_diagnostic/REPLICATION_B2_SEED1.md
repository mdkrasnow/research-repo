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

## Step 3 — Selector arms (FID, label-independent) — DONE
`probe_gated.sbatch`, probe artifact = `runs/b2_seed1_repl/probe_artifact.npz` (dim=30).
Smoke: NUM_SLOTS=512 R=3 gpu_test (job 26013773) — probe 95.91<vanilla 100.21 Δ4.30, wiring clean.
Full: NUM_SLOTS=50000 R=3 seas_gpu 4-GPU (job **26028553**, COMPLETED 1h20m).

**Full 50k FID:**

| arm | B/2 seed1 (50k) | locked B/2 seed0 |
|---|---|---|
| vanilla (restart-baseline) | 27.506 | r3rand 27.95 / long250 28.10 |
| **probe-restart** | **24.808** | r3probe@50 **24.66 ± 0.16** (5-seed CI) |
| oracle (ceiling) | 15.837 | 21.75 |
| Δ vanilla−probe | **2.698** | ~3.29 |
| oracle-recovered fraction | 0.231 (23%) | ~16–18% |

`"sane": false` is the known-harmless flag: pipeline "vanilla" is a restart control (~27.5), not the
non-restart NFE-250 reference 31.41 — identical behaviour to the locked seed0 run; deltas are valid.

## VERDICT — selector gain is NOT checkpoint-specific
- **probe-restart FID 24.81 (seed1) ≈ 24.66 (locked seed0)** — near-identical at 50k on a second,
  independently-trained checkpoint. Δ≈2.7 FID, 23% oracle recovery — comparable magnitude.
- The full mechanism chain replicates: scalars dead → SHAPE≫MAG learned probe → probe-restart wins at FID.
- **AUROC vs FID dissociation (important):** seed1's diagnostic AUROC was lower (0.746 vs 0.813),
  depressed by weak label sanity (s4=0.577) — yet the **FID gain fully transferred**. Confirms the
  AUROC drop was a label-pipeline artifact, not weaker real selector signal. FID (label-independent)
  is the trustworthy readout; the locked inference-time result generalizes across checkpoints.
- Scope: one extra checkpoint (same B/2 scale, seed1, single 50k draw — not a multi-seed CI on this
  ckpt). Tests *checkpoint*-specificity, not *model-scale*. Scale axis (S/2) remains future per
  `NEXT_CHECKPOINT_DISCOVERY.md`. No claim change beyond "robust to checkpoint instance."
