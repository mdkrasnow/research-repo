# Experiment 1 — NFE / sampler robustness (vanilla EqM vs ANM EqM v10)

**Evaluation only.** Compares two FROZEN EqM-B/2 IN-1K checkpoints under a sampler
sweep. No training code is touched.

- **vanilla**: `stage_b_vanilla_in1k_80ep_seed0/.../checkpoints/0380000.pt` (FID 31.41)
- **anm (v10)**: `imagenet1k_80ep_v10_seed0/.../checkpoints/final.pt` (λ=0.1, FID 29.01)

Grid = 80 cells: `{vanilla,anm} × {gd,ngd} × nfe{10,25,50,100,250} × step_mult{0.5,1.0,1.5,2.0}`.

## Files
| File | Role |
|---|---|
| `sample_gd_fixed.py` | Deterministic fork of `eqm-upstream/sample_gd.py`: fixed init latents + fixed balanced labels (indexed by global sample id), `--step-mult`, writes `gen_stats.json`. EMA/VAE/CFG/loop identical to upstream. |
| `compute_metrics.py` | FID (frozen ref mu/σ) + KID (poly-kernel MMD, degree 3) from cached Inception-2048 features; bootstrap CIs. |
| `exp1_driver.py` | Orchestrator: precompute fixed inputs + frozen reference once, 80-cell loop, resume, CSV/manifest, then analysis + plots. |
| `analysis.py` | log/linear-NFE AUC, low-NFE & stress scores, nfe-to-match, pairwise delta. |
| `plots.py` | FID/KID-vs-NFE, ANM−vanilla heatmaps, AUC/low-NFE/stress bars, nfe-to-match, wall-clock Pareto. |
| `../../configs/exp1_eval.yaml` | Pre-registered settings (base_step 0.003, mu 0.3, cfg 1.0, vae ema). |
| `../../slurm/jobs/exp1_sweep.sbatch` | Cluster wrapper (clone-at-SHA, run grid on 4×A100 seas_gpu, sync metrics/plots back). |

## Run (cluster, GPU-only — submit via `scripts/cluster/remote_submit.sh`)

```bash
# Dry-run (validates ckpts/ref, prints 80 run_ids; writes nothing)
DRY_RUN=1 VANILLA_CKPT=<v.pt> ANM_CKPT=<a.pt> sbatch projects/diff-EqM/slurm/jobs/exp1_sweep.sbatch

# Smoke A (2 cells, 64 samples): pipeline correctness
SMOKE=A NUM_SAMPLES=64 NUM_GPUS=1 VANILLA_CKPT=<v.pt> ANM_CKPT=<a.pt> sbatch .../exp1_sweep.sbatch

# Smoke B (16 cells, 256 samples): gd+ngd, stress step2.0, heatmap/AUC plumbing
SMOKE=B NUM_SAMPLES=256 NUM_GPUS=1 VANILLA_CKPT=<v.pt> ANM_CKPT=<a.pt> sbatch .../exp1_sweep.sbatch

# Pilot (80 cells, 5k): catastrophic-instability + runtime estimate ONLY (do not tune on it)
NUM_SAMPLES=5000 VANILLA_CKPT=<v.pt> ANM_CKPT=<a.pt> sbatch .../exp1_sweep.sbatch

# Full (80 cells, 50k, 4 GPU)
VANILLA_CKPT=<v.pt> ANM_CKPT=<a.pt> sbatch .../exp1_sweep.sbatch

# Resume (skip cells with samples+features+metric) / plots-only (rebuild from results.csv)
RESUME=1 ... sbatch .../exp1_sweep.sbatch
PLOTS_ONLY=1 ... sbatch .../exp1_sweep.sbatch
```

Local use is `--dry-run` / `--plots-only` only (sampling needs a GPU + the cluster ImageNet ref).

## Compute estimate
Cost is dominated by high-NFE cells. Per arm, one sampler: Σ(nfe−1) over {10,25,50,100,250} ≈ 430 field evals × 50k samples. The 250-step cells (×2 samplers ×4 step_mults ×2 arms = 16 cells) carry most of the cost; the nfe≤50 cells are cheap. Rough order: **~half a day on 4×A100** for the full 80-cell 50k sweep (FID/KID extraction adds ~minutes/cell). Confirm with the 5k pilot before committing the full run.

## Output schema
- `metrics/results.csv` — one row/cell (full schema: run_id, checkpoint_type/path/sha, sampler, nfe, step_mult, base/step_size, nag_mu, fid + CI, kid mean/std + CI, wall_clock, images_per_sec, nfe_field=nfe−1, nfe_raw_forward, nan/divergence/clip, latent-norm + grad-norm stats, feature_path, git_commit, timestamp, notes).
- `metrics/auc_summary.csv` — per (ckpt,sampler,step_mult) AUC + a `step_mult=ALL` global/stress row.
- `metrics/pairwise_delta.csv` — anm−vanilla FID/KID per matched cell + win flags.
- `metrics/nfe_to_match.csv` — min ANM NFE to reach vanilla 250-step-default and vanilla best-grid FID.
- `metrics/run_manifest.jsonl` — per run: ckpt sha, config hash, fixed-latent hash, step_size, png count.
- `plots/` — FID/KID-vs-NFE (log-x, faceted), ANM−vanilla FID/KID heatmaps (per sampler), AUC bar, low-NFE & stress bars, nfe-to-match, wall-clock-vs-FID.

## Interpreting success / failure (pre-registered)
**ANM supported** if most hold: lower global log-NFE FID AUC; lower mean FID over nfe∈{10,25,50}; reaches vanilla's 250-step/default FID at fewer NFEs; no collapse at low-NFE/high-step (nan/divergence ~0, KID not blown up); KID agrees directionally with FID.

**Failure** if: ANM wins only at one tuned cell; loses on AUC despite winning at 250; collapses at nfe∈{10,25} or step_mult∈{1.5,2.0}; better default FID but worse *relative* degradation; needs per-checkpoint step retuning; or FID improves while KID / visual grids show diversity collapse.

**Ambiguous** if: FID improves but KID worsens; gains only under one sampler; deltas within bootstrap CIs.

## Known caveats (honest reporting)
1. **`step_mult` couples into the time schedule** (`t += stepsize`): it scales both the GD step and the γ/t trajectory. Label the axis "step+time mult", not pure step size.
2. **`nfe_field = nfe − 1`** — upstream loop runs `num_sampling_steps−1` evals; kept for baseline reproducibility, reported truthfully.
3. **Reference is frozen + deterministically ordered** (no `shuf`) and reused by all 80 cells — required for cross-condition FID comparability.
