# Experiment 4 — Seed stability, train-val gap, memorization audit

Vanilla EqM vs ANM-EqM (v10 PGD hard-example mining). Tests whether ANM's FID
gain is genuine generalization / field robustness rather than overfitting,
memorization, duplication, or one lucky seed. Design + rationale:
`../../documentation/experiment-4-agent-prompt.md`.

## Two load-bearing caveats (do not ignore)
- **B1**: the project's trusted FIDs (31.41 / 29.01 / 27.09) are FID-vs-**TRAIN**
  (`slurm/jobs/imagenet1k_fid_eval.sbatch:53`). This experiment computes BOTH
  val- and train-reference FID/KID through one unified `{mu,sigma}` evaluator
  (`build_references.py`). `FID_gap = val − train` is the new quantity.
- **B2**: only single-seed checkpoints exist today. Until ≥3 seeds per arm land
  (Phase 2), this is a **checkpoint variability audit, NOT a seed-stability
  audit**. The driver prints this banner and sets `seed_audit_valid=false` in
  `aggregate_summary.csv`. Do not claim seed stability while that is false.

## Files
| File | Role |
|---|---|
| `features.py` | inception / dinov2 / clip / lpips / **stub** (test-only) extractors |
| `metrics.py` | FID (reuses `evaluate_fid.py`) + KID (new, unbiased poly-MMD) + bootstrap CI |
| `nn_audit.py` | tiled-GPU cosine NN, memorization stats, duplicate clustering |
| `plots.py` | seed error bars, train-val gap, NN-ratio hist, dup rate, step-vs-compute, NN panels |
| `eval_stability_memorization.py` | driver (config → per-seed CSV + aggregate + JSON + panels + plots) |
| `build_references.py` | builds val+train inception stats, KID feats, DINO NN banks (equal counts) |
| `local_smoke.py` | CPU plumbing smoke (stub backbone, zero deps) — `python local_smoke.py` |
| `configs/full_example.json` | template config (replace REPLACE paths) |

## Run order (cluster, GPU; SLURM is remote-only via `scripts/cluster/*`)
```bash
# 0. plumbing check (local, CPU)
python projects/diff-EqM/experiments/experiment4/local_smoke.py

# 1. cluster smoke (real inception+dino weights/deps)
bash scripts/cluster/remote_submit.sh projects/diff-EqM/slurm/jobs/exp4_smoke.sbatch diff-EqM

# 2. build references ONCE (fixes B1: val + class-balanced train, equal counts)
bash scripts/cluster/remote_submit.sh projects/diff-EqM/slurm/jobs/exp4_build_refs.sbatch diff-EqM

# 3. generate samples per checkpoint (FIXED protocol + label schedule).
#    Same GLOBAL_SEED + LABEL_SEED + world_size for EVERY checkpoint.
CHECKPOINT_PATH=<vanilla 0380000.pt> OUT_NPZ=projects/diff-EqM/results/experiment4/samples/vanilla_step_seed0.npz \
  bash scripts/cluster/remote_submit.sh projects/diff-EqM/slurm/jobs/exp4_generate.sbatch diff-EqM
CHECKPOINT_PATH=<v10 0380000.pt>     OUT_NPZ=.../anm_step_seed0.npz \
  bash scripts/cluster/remote_submit.sh projects/diff-EqM/slurm/jobs/exp4_generate.sbatch diff-EqM
#   compute-matched: see below.

# 4. fill configs/full_example.json samples_npz paths, then run the audit
CONFIG=projects/diff-EqM/experiments/experiment4/configs/full_example.json \
  bash scripts/cluster/remote_submit.sh projects/diff-EqM/slurm/jobs/exp4_audit.sbatch diff-EqM
```

## Compute-matched (MANDATORY — do not skip)
ANM does extra inner forwards. Step-matched alone cannot support a fair
generalization claim. Two budgets:
1. **ANM-final**: v10@380K vs vanilla at matched FEU (~760K steps). That vanilla
   checkpoint **does not exist** → launch an extended-vanilla train first.
2. **Vanilla-final**: vanilla@380K vs v10@~190K (verify the early v10 ckpt
   survived the pruner; else re-emit).

Exact compute accounting: `train_imagenet.py` now persists
`anm_inner_forward_count` in every checkpoint (K inner + 1 hard forward per
mined step, resume-safe). Vanilla = 0. For pre-existing checkpoints without the
counter, use the ~2.0× FEU estimate and label the comparison approximate. If no
compute-matched checkpoint is available, the driver emits a
`BLOCKED_NEED_CKPT` row rather than silently dropping the regime.

## Known limitation
Class-conditional NN needs per-sample labels aligned to the generated `.npz`.
DDP label-saving from `sample_gd.py` is a follow-up; until then set
`class_conditional_nn:false` (or omit labels) and the audit uses global NN,
logging that it did so.

## Outputs (`results/experiment4/`)
`per_seed_results.csv`, `aggregate_summary.csv` (mean/std/sem/median/95% CI +
paired deltas + `seed_audit_valid`), `nn_stats/**.json`, `nn_panels/**.png`,
`plots/*.png`, `AUDIT_STATUS.txt`.
