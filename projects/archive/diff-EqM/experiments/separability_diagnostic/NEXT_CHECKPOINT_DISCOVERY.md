# NEXT_CHECKPOINT_DISCOVERY — second EqM checkpoint for selector replication

**Date:** 2026-06-29
**Task:** find a second EqM checkpoint/model to re-run the locked B/2 metacognition
selector protocol (arms: `long250`, `r3rand`, `r3energy`, `r3probe@50`, optional
`oracle`/headroom) on — to test whether the result is checkpoint/model-specific.
**Scope of this doc:** DISCOVERY ONLY. No jobs launched, no sampling, no probe trained,
no FID computed. Commands below are for LATER. See **DO NOT RUN YET** at bottom.

---

## Executive recommendation

**Primary (single best next checkpoint): `imagenet1k_80ep_vanilla_eqm-b-2_seed1` — B/2 seed1, vanilla, COMPLETE (`final.pt`).**

Why it wins the "is the result checkpoint-specific?" question with lowest risk:
- **Identical harness, identical arch** (`EqM-B/2`, `uncond=True, ebm="none"`, num_classes=1000,
  VAE `sd-vae-ft-ema`). The locked B/2 sampler config — `gd`, `eta/stepsize=0.003`,
  `steps=250`, `cfg=1.0` — applies *verbatim*. Zero loader risk, zero re-tuning.
- **Directly comparable FID lineage.** Same scale → its baseline FID lands on the
  trusted ~27.93/31.41 B/2 line; the probe−random gap is read on the same axis as the
  locked 26.90 vs 27.89 result. A different seed isolates the "different checkpoint
  instance, same everything else" axis — the cleanest definition of "checkpoint-specific."
- **Training complete** (`final.pt`, step ≈400000 / 80ep).

**Strong second (scale axis): `imagenet1k_80ep_vanilla_eqm-s-2_seed0` — S/2 seed0, vanilla, COMPLETE (`final.pt`).**
Tests *model-scale* dependence (the harder, more novel question). Same loader (just
`MODEL=EqM-S/2`), cheapest to sample (527 MB ckpt). **Caveat:** the locked `eta=0.003`
stepsize was tuned for B/2 per the repo eval table — it is *unverified* at S/2, so a
load-only + tiny-sample sanity is required before trusting selector numbers. Run AFTER
seed1.

**Do NOT use L/2 (see table): undertrained, single stalled checkpoint.**

Suggested plan: run seed1 first (answers checkpoint-specificity at zero risk), then S/2
(answers scale-specificity). Together they bracket both axes of "is this result special
to one model."

---

## Ranked candidate table

All paths under `…/research-repo/projects/diff-EqM/results/` on cluster (`/n/home03/mkrasnow/…`).
Probe artifact = NONE exists for any of these (the only existing
`runs/b2_vanilla/probe_artifact.npz` belongs to the locked stage_b ckpt). All require a
fresh sample→probe pass (see commands). FID ref stats present for all (shared).

| Rank | Candidate | Scale | Seed | ckpt file | size | complete? | loader change | config risk | why |
|---|---|---|---|---|---|---|---|---|---|
| **1** | `imagenet1k_80ep_vanilla_eqm-b-2_seed1` | B/2 | 1 | `…/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/final.pt` | 2.09 GB | ✅ final.pt | none | **none** (locked B/2 config exact) | best: isolates checkpoint/seed, FID directly comparable |
| **2** | `imagenet1k_80ep_vanilla_eqm-s-2_seed0` | S/2 | 0 | `…/000-EqM-S-2-Linear-velocity-None-vanilla/checkpoints/final.pt` | 527 MB | ✅ final.pt | `MODEL=EqM-S/2` only | low: eta=0.003 untuned at S/2 → load+tiny-sample sanity first | tests model-scale dependence; cheapest |
| 3 | `imagenet1k_80ep_vanilla_eqm-b-2_seed2` | B/2 | 2 | `…/checkpoints/0365000.pt` | 2.09 GB | ⚠ step 365k (~73ep, no final.pt) | none | none | 2nd seed replica; slightly under locked 380k/76ep — usable, marginally less trained |
| 4 | `imagenet1k_80ep_v10_s2_seed0` | S/2 | 0 | `…/000-EqM-S-2-…-dganm/checkpoints/final.pt` | 527 MB | ✅ | `MODEL=EqM-S/2` | low | ANM (v10), not vanilla — use only if testing selector on the *treated* model |
| 5 | `imagenet1k_80ep_v10_b2_seed1` / `_seed2` / `_lambda03_seed0` / `_k3_seed0` | B/2 | 1/2/0 | (ANM dganm dirs) | ~2.1 GB | mixed | none | none | v10 ANM variants; off-axis for a *vanilla* replication |
| ✗ | `imagenet1k_80ep_vanilla_eqm-l-2_seed0` | L/2 | 0 | `…/000-EqM-L-2-…/checkpoints/0080000.pt` | 7.33 GB | ❌ **single ckpt, step 80000 (~16ep of 80), stalled 2026-05-30** | none | **HIGH** — undertrained → poor FID confounds selector signal | **REJECT** unless retrained |
| — | locked reference (not a candidate) | B/2 | 0 | `stage_b_vanilla_in1k_80ep_seed0/…/checkpoints/0380000.pt` | 2.09 GB | trusted (FID 31.41) | — | — | the original; has `probe_artifact.npz` |

CIFAR / maze / MNIST / Sudoku EqM checkpoints exist in repo but are **excluded per task**
(need another image-generation IN-1K EqM model; only fall back if no IN-1K existed — it does).

---

## Exact paths (cluster, absolute)

```
ROOT=/n/home03/mkrasnow/research-repo/projects/diff-EqM/results
# Rank 1  B/2 seed1 (vanilla, COMPLETE)
$ROOT/imagenet1k_80ep_vanilla_eqm-b-2_seed1/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/final.pt
# Rank 2  S/2 seed0 (vanilla, COMPLETE)
$ROOT/imagenet1k_80ep_vanilla_eqm-s-2_seed0/000-EqM-S-2-Linear-velocity-None-vanilla/checkpoints/final.pt
# Rank 3  B/2 seed2 (vanilla, step 365k)
$ROOT/imagenet1k_80ep_vanilla_eqm-b-2_seed2/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0365000.pt
# REJECT  L/2 seed0 (undertrained)
$ROOT/imagenet1k_80ep_vanilla_eqm-l-2_seed0/000-EqM-L-2-Linear-velocity-None-vanilla/checkpoints/0080000.pt
```

FID reference stats (present, shared):
```
in1k_reference_stats.npz                                  # used by the locked SCALE_RESULTS 50k runs
/n/holylabs/ydu_lab/Lab/raywang4/VIRTUAL_imagenet256_labeled.npz
/n/holylabs/ydu_lab/Lab/hlillemark/projects/EqM/VIRTUAL_imagenet256_labeled.npz
/n/holylabs/ydu_lab/Lab/raywang4/imagenet/{train,val}     # real-image dirs for quality labels
```

---

## Compatibility analysis (per the locked harness)

`sample_with_logging.py` builds the model generically:
`EqM_models[args.model](input_size, num_classes=args.num_classes, uncond=True, ebm="none")`
then loads `state["ema"]` → `state["model"]` → raw. **Any standard EqM size checkpoint
(S/2, B/2, L/2, XL/2) loads with no code change — only `--model` differs.** VAE and
num_classes=1000 are fixed and match all IN-1K candidates above.

`slurm/jobs/sep_diag.sbatch` is fully parameterized by `CKPT_PATH` + `MODEL` env vars
(it even auto-resolves a dir to the latest `*.pt`). Replication = set two env vars.
Stages: (1) `sample_with_logging.py` emits `logs/traj_rank*.npz` + decoded PNGs →
(2) `compute_quality_labels.py` → (3) `compute_scores.py` → (4) `analyze.py`.
`learned_probe.py --folder <OUT_DIR>` trains the probe from the stage-1 logs.

**Conditioning/VAE assumptions match B/2 harness for all IN-1K candidates** (class-cond
1000, sd-vae-ft-ema, 256px, latent 32). Only open compatibility item: stepsize `eta=0.003`
is B/2-specific → for S/2 (and L/2) verify via load + ≤200-sample sanity before trusting.

---

## Missing dependencies / artifacts (per candidate)

- **Probe artifact:** NONE for any candidate. Must be generated fresh (sample→probe). The
  existing `probe_artifact.npz` is bound to the locked stage_b B/2 ckpt and is not transferable.
- **Trajectory logs:** NONE for any candidate (no `logs/traj_rank*.npz` outside `runs/b2_vanilla`).
  Generated by stage-1 sampling — they don't pre-exist; producing them IS the first run.
- **FID ref:** present (above) — no missing dep.
- **Known baseline FID for S/2 / L/2 / B/2-seed1:** not numerically recorded in repo
  (SCALE_RESULTS documents only the B/2-seed0 50k lineage; S/2-L/2 Phase-3 numbers were
  "figure-only" / pending). The replication run produces its own baseline arm (`long250`),
  so this is not blocking — but there is no pre-existing trusted FID to cross-check against.

---

## Commands needed LATER — VALIDATE LOAD ONLY (cheap, do these first)

Confirms the ckpt loads into the locked harness and (for S/2) that sampling is sane.
Run on `gpu_test` single-GPU (`--nproc_per_node=1`, MIG-safe). **Tiny sample count.**

```bash
# load-only smoke (B/2 seed1) — 64 samples, no FID, just "does it run + look sane"
CKPT_PATH="projects/diff-EqM/results/imagenet1k_80ep_vanilla_eqm-b-2_seed1/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/final.pt" \
MODEL="EqM-B/2" NUM_SAMPLES=64 \
  sbatch projects/diff-EqM/slurm/jobs/sep_diag.sbatch   # 4-GPU; or convert to --nproc_per_node=1 for gpu_test

# S/2 sanity (verify eta=0.003 transfers) — 200 samples, eyeball PNGs in OUT_DIR
CKPT_PATH="projects/diff-EqM/results/imagenet1k_80ep_vanilla_eqm-s-2_seed0/000-EqM-S-2-Linear-velocity-None-vanilla/checkpoints/final.pt" \
MODEL="EqM-S/2" NUM_SAMPLES=200 \
  sbatch projects/diff-EqM/slurm/jobs/sep_diag.sbatch
```

## Commands needed LATER — GENERATE PROBE + RUN SELECTOR (the actual replication)

```bash
# 1. Full sample+log pass (produces logs/traj_rank*.npz + PNGs). NUM_SAMPLES per locked protocol.
CKPT_PATH="<candidate ckpt path>" MODEL="<EqM-B/2 | EqM-S/2>" NUM_SAMPLES=<locked N> \
  sbatch projects/diff-EqM/slurm/jobs/sep_diag.sbatch

# 2. Train probe over that run's folder (5-fold, matches locked probe)
python projects/diff-EqM/experiments/separability_diagnostic/learned_probe.py \
  --folder <OUT_DIR> --folds 5 --seed 0 --l2 1.0 --n-bins 5

# 3. Selector arms (long250 / r3rand / r3energy / r3probe@50 / oracle) — reuse the locked
#    selector driver (probe_gated_sample.py / selector_compare.py) pointed at the new
#    ckpt + new probe artifact, same eta=0.003/250/cfg1.0 config as the locked B/2 run.
```

(Exact selector-arm invocation = mirror the locked B/2 SELECTOR_LOCKDOWN_RESULTS.md run,
swapping ckpt + probe artifact paths. Confirm arm definitions against that file before launch.)

---

## ⚠️ DO NOT RUN YET

This is a discovery report. **Do NOT** submit SLURM jobs, sample, train probes, or compute
FID off this doc alone. Before any compute: (1) reconcile `pipeline.json:active_runs` vs
`squeue`, (2) confirm partition per CLAUDE.md SLURM discipline, (3) get explicit go-ahead on
which candidate(s). Recommended order when greenlit: **B/2 seed1 (load-only → full) → S/2 seed0**.

## Uncertainty / conservative assumptions logged
- S/2 (and L/2) stepsize `eta=0.003` is B/2-tuned; assumed transferable but **unverified** → load+tiny-sample gate required.
- No pre-existing trusted baseline FID for S/2/L/2/B/2-seed1 → replication self-baselines via `long250`; no external cross-check.
- B/2 seed2 has no `final.pt` (step 365k ≈73ep) → treated as near-complete, ranked below seed1's `final.pt`.
- L/2 single stalled ckpt (step 80000) assumed undertrained/untrustworthy → rejected, not investigated further.
- Holylabs `…/Lab/mkrasnow/` is empty; all candidate ckpts live on `/n/home03`. No holylabs ckpt mirror found.
