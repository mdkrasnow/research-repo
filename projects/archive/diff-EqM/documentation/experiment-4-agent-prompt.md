# Experiment 4 — Coding-Agent Implementation Prompt

**Audience:** a coding agent with repo access (rg/find/ls/Read/Edit/Bash).
**Task:** implement the seed-stability / train-val generalization-gap / memorization audit comparing **vanilla EqM** vs **ANM-EqM (v10 PGD hard-example mining)**.
**Scope (HARD):** touch only `projects/diff-EqM/**`, `scripts/cluster/**`. Do not read or modify other projects.

---

## 0. Read before you act (do not skip)

A scouting pass already mapped this repo. Treat the map in §2 as a starting set of leads — **verify each path with Read before depending on it** (line numbers drift). Two findings are load-bearing and change the design; both are documented below. Do not "simplify" them away.

### BLOCKER B1 — trusted FIDs are TRAIN-reference, not VAL-reference
The production FID path `slurm/jobs/imagenet1k_fid_eval.sbatch` sets `IMAGENET_REF=.../imagenet/train` and recomputes Inception activations from a flat symlink dir of 50K subsampled **training** images. So the trusted numbers **FID 31.41 (vanilla) / 29.01 (v10 λ=0.1) / 27.09 (v10 λ=0.3) are all FID-vs-TRAIN**, single seed each. A separate sbatch `compute_in1k_reference_stats.sbatch` builds a **val** `.npz` (`mu,sigma,num_images`) but the production path never consumes a `.npz` (it recomputes), and the two formats diverged — that is the `KeyError: mu` mismatch noted in `documentation/debugging.md`. **The project has never computed a val-reference FID.** Experiment 4's central quantity `FID_gap = FID_val − FID_train` therefore requires you to (a) build/cache a val-reference stat set, (b) build a fixed class-balanced 50K train-reference subset, (c) score both through ONE unified evaluator with ONE Inception extractor and ONE `.npz` schema `{mu, sigma}`. This is new code, not reuse.

### BLOCKER B2 — only single-seed checkpoints exist today
On disk: vanilla seed0, v10 λ=0.1 seed0, v10 λ=0.3 seed0 — **one seed each**. Phase 2 seeds 1+2 are queued/running, not landed. Per the experiment's own rules, a one-seed-only result is a **failure for the seed-stability claim** and must be labelled a `checkpoint variability audit`, NOT a seed audit. **Mandate:** build the full harness now (it is fully reusable and consumes new seeds as they land), run the smoke + the single-seed variability pass, but **emit no seed-stability conclusion** until ≥3 seeds per arm exist. Surface this caveat in every aggregate artifact (a `regime`/`n_seeds` field and a printed warning).

### Compute-matched is mandatory
ANM does extra inner-loop forwards. Step-matched alone cannot support a fair generalization claim. You MUST implement and run both regimes (see §4). If a compute-matched checkpoint is unavailable, you must say so explicitly and state exactly which checkpoint/train job is needed — never silently drop the regime.

---

## 1. Phase plan (execute in order; stop for approval after Phase 1 if anything contradicts §2)

- **Phase 1 — Discovery.** Verify the §2 file/function map against the live repo. Return a corrected map (path:line — what — reuse) before writing code. Confirm: ckpt naming, sampler protocol, EMA load, Inception extractor, train/val reference availability, val split presence on cluster, installed deps (faiss/clip/lpips/dinov2/torchmetrics).
- **Phase 2 — Audit harness.** Add `projects/diff-EqM/experiments/experiment4/eval_stability_memorization.py` + a small package (`__init__.py`, `features.py`, `nn_audit.py`, `metrics.py`, `plots.py`). Driven by a JSON/YAML config mapping `regime -> checkpoint_type -> seed -> checkpoint_path`. Fixed eval seeds + fixed label schedule + identical sampler/NFE/stepsize/EMA/CFG/VAE/sample-count across all checkpoints.
- **Phase 3 — Metrics.** Per checkpoint: generate (or load cached) samples → Inception features → FID+KID vs val-ref AND train-ref → gaps → DINO/CLIP NN vs train & val banks → memorization stats → duplicate stats → suspicious-panel metadata.
- **Phase 4 — Outputs.** Per-seed CSV, aggregate CSV (mean/std/sem/median/95% bootstrap CI + paired deltas), NN-stats JSON, NN panels, plots.
- **Phase 5 — Smoke.** Tiny config (N≤1000, ref≤5K, NN bank≤10K), 1 vanilla + 1 ANM ckpt (or two fake checkpoints). Verify the whole pipeline end-to-end incl. an intentionally inserted duplicate being detected, and reproducibility (same eval seeds → identical features within tolerance). Smoke FID/KID are NOT scientific evidence.
- **Phase 6 — Full run.** Provide exact step-matched and compute-matched commands, the checkpoint-config build steps, and failure detection. Run via the cluster (GPU); SLURM is remote-only through `scripts/cluster/*`.

---

## 2. Verified starting map (confirm before use)

### Training & checkpoints
- `experiments/train_imagenet.py:905-918` — ckpt save `{results_dir}/checkpoints/{train_steps:07d}.pt`, dict `{model,ema,opt,args,train_steps,epoch}`. **`train_steps`+`epoch` are in the ckpt → step-matched selection is exact.**
- `experiments/train_imagenet.py:509-512` — seed `global_seed*world+rank`.
- `experiments/train_imagenet.py:403-452` — `_v10_pgd_hard_example_step`: `model.eval()` at :426, `x_t.detach()` at :433, final `δ.detach()` at :440, K=1 sign-PGD. **Inner forward COUNT is not logged** (only `v10_hard_loss`, `aux/base`, `||δ||` at :807-868).
- `slurm/jobs/imagenet1k_80ep_v10.sbatch` (flags `--gamma 0.1 --v10-K 1 --v10-eps-radius 0.3 --mining-lr 0.05 --mine-every 1`) and `imagenet1k_80ep_vanilla.sbatch` — the two train wrappers.
- `configs/cafm/eqm_b2_in256_cafm.yaml` — EqM-B/2, 256, 1000-class, λ=0.1, ckpt interval 5000.

### Sampling (the fixed eval protocol)
- `eqm-upstream/sample_gd.py:186-207` — GD/NAG sampler. FID protocol: `--sampler gd --stepsize 0.003 --num-sampling-steps 250 --cfg-scale 1.0`.
- `sample_gd.py:174` — labels `torch.randint(0,num_classes,(n,))` — **random, no fixed schedule. You MUST add a seeded label generator** (e.g. `--label-seed`) so the label schedule is identical across checkpoints.
- `sample_gd.py:92-95` — per-rank seed `global_seed*world+rank`; deterministic only if `global_seed` AND `world_size` are pinned. Pin both for the audit.
- `sample_gd.py:109-122,152` — loads and samples from **EMA** weights (consistent EMA policy already).
- `sample_gd.py:43-57,203-206` — saves `{i:06d}.png` and `{folder}.npz` (key `arr_0`, uint8 NHWC).

### FID / KID
- `experiments/evaluate_fid.py:40-78` — `InceptionV3Features` (torchvision, 2048-d pool3, bilinear→299). **Use this as the single unified extractor for FID/KID.**
- `experiments/evaluate_fid.py:85-111` — `compute_statistics`, `compute_fid` (scipy sqrtm). Reuse for both references.
- `slurm/jobs/compute_in1k_reference_stats.sbatch:25,70` — builds val `.npz` `{mu,sigma,num_images}` from `.../imagenet/val` (50/class×1000). Run/adapt for val-ref; build a parallel train-ref subset the same way.
- **KID: absent.** Implement (unbiased polynomial-kernel MMD over Inception features, or torchmetrics `KernelInceptionDistance` — present only in `external/Adversarial-Flow-Models` env; prefer a self-contained numpy/torch impl to avoid the dep).

### Dataset
- Train: `experiments/train_imagenet.py:603-619` `ImageFolder` @ `/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train`.
- Val: `experiments/eval_capabilities.py:150-174,392` @ `/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val` (**val split exists on cluster**). Reuse/extend its synset-ordered loader; for the NN/FID banks you need full-coverage loaders, not one-per-class.
- VAE: `train_imagenet.py:587,665` SD-VAE-ft-ema, latent `*0.18215`, encode at runtime. Transforms `:104-117,598-601` ADM center-crop 256 + hflip + normalize[-1,1]. **For reference banks, match preprocessing to whatever the extractor expects (Inception: its own resize/normalize; DINO/CLIP: their own). Do NOT feed [-1,1] EqM latents to Inception.**

### Feature / NN / viz — mostly absent (add)
- DINOv2: NOT FOUND → add via `torch.hub.load('facebookresearch/dinov2','dinov2_vitl14')`, L2-normalize, cosine. Primary NN feature.
- CLIP: only `external/.../common/metrics.py:229-253`. Add a self-contained CLIP (open_clip) for the semantic cross-check.
- LPIPS: optional import `eval_capabilities.py:186-192` (net alex), not in env → install for top-suspicious-pair visual-copy check only.
- FAISS: NOT FOUND. At 50K queries × (≤1.28M train) banks, use **tiled GPU `torch.cdist`** (chunk queries, keep running top-k). Add FAISS only if cdist is too slow; gate behind a try-import.
- Panels: `torchvision.utils.make_grid/save_image` (`eval_capabilities.py:44,378`).

### Distributed / logging / conventions / cluster
- DDP sampling `sample_gd.py:87-96`. Run-folder pattern `train_imagenet.py:519-526`.
- Results aggregations `results_variants.tsv`; per-run `train_log.tsv` (`dganm_variants/_common.py:305`).
- Cluster helpers all present: `scripts/cluster/{ssh,status,remote_submit,remote_fetch,ensure_session}.sh`. sbatch template = clone→`/tmp`, checkout `GIT_SHA`, pip, torchrun, rsync back (model on `slurm/jobs/cafm_eqm_smoke.sbatch`). **SLURM is remote-only.**

---

## 3. Checkpoint selection

### Step-matched
- Anchor = 380000 steps (80ep). Select vanilla `…/checkpoints/0380000.pt` and v10 `0380000.pt`. Tolerance 0 (exact anchor exists). Missing → nearest by `train_steps`, report the mismatch in the CSV.

### Compute-matched (mandatory)
- **Add a forward counter** to the v10 step: accumulate `anm_inner_forward_count += K (inner grad fwds) + 1 (hard fwd)` per mined step and persist into the ckpt dict next to `train_steps` (`train_imagenet.py:444` increment, `:910` save). Vanilla counter = 0. Going forward (Phase 2 seeds) this gives exact counts.
- **Existing single-seed ckpts have no counter** → use the FEU estimate and label the comparison approximate (the ≤5% match rule cannot be met without counts):
  - `FEU_vanilla/step ≈ 3` (fwd+bwd).
  - `FEU_v10/step ≈ 6` (outer 3 + inner grad ~2 + hard fwd 1) ⇒ **≈2.0× vanilla** (conservative; ~1.7× if δ-grad < full bwd).
- Two compute-matched targets (implement both selectors; run whichever has checkpoints, log which):
  1. *ANM-final budget*: v10@380K vs vanilla trained to matched FEU (**~760K steps — checkpoint does NOT exist; requires a new extended-vanilla train ~24h gpu**).
  2. *Vanilla-final budget*: vanilla@380K vs v10@~190K (**may be pruned** — pruner keeps {5000,65000}+latest2; verify on disk, else re-emit).
- **If neither compute-matched ckpt is available: do NOT skip the regime.** Emit a stub row with `status=BLOCKED_NEED_CKPT` and print the exact train job to launch.

Config drives selection:
```json
{
  "sampler": {"name":"gd","stepsize":0.003,"num_sampling_steps":250,"cfg_scale":1.0,"use_ema":true},
  "eval": {"n_eval":50000,"global_seed":0,"world_size":4,"label_seed":12345,"num_classes":1000},
  "references": {
    "val_stats_npz":  "projects/diff-EqM/results/in1k_val_ref_stats.npz",
    "train_stats_npz":"projects/diff-EqM/results/in1k_train_ref_stats.npz",
    "val_nn_bank":    "projects/diff-EqM/results/in1k_val_dino.npy",
    "train_nn_bank":  "projects/diff-EqM/results/in1k_train_dino.npy"
  },
  "feature_backbone_fid":"inception_pool3",
  "feature_backbone_nn":"dinov2_vitl14",
  "checkpoints": {
    "step_matched": {
      "vanilla": {"0":"…/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt"},
      "anm":     {"0":"…/v10_lambda01_seed0/…/checkpoints/0380000.pt"}
    },
    "compute_matched": {
      "vanilla": {"0":"…/0380000.pt"},
      "anm":     {"0":"…/0190000.pt"}
    }
  }
}
```

---

## 4. Metrics (per checkpoint)

1. Generate N=50K with the fixed sampler + fixed `label_seed` + pinned `global_seed`/`world_size` (reuse `sample_gd.py`; add `--label-seed`). Cache the `.npz`.
2. Inception pool3 features (reuse `InceptionV3Features`).
3. `FID_val`, `KID_val` vs val-ref; `FID_train`, `KID_train` vs train-ref. Equal 50K/50K reference counts.
4. `fid_gap = FID_val − FID_train`; `kid_gap = KID_val − KID_train`.
5. DINOv2 features for generated + banks (L2-norm, cosine). NN vs train bank and val bank, top-3. Also class-conditional NN (same-class only) — required on IN-1K.
6. Memorization: `mean/median d_train`, `mean/median d_val`, `ratio=d_train/(d_val+eps)`, `margin=d_val−d_train`, `frac_ratio<1.0/0.9/0.8`, percentiles, `frac d_train<τ_mem`.
7. Duplicates: self-NN in DINO space, `τ_dup` = 0.1% quantile of val→train NN distances; near-dup rate, cluster count, largest cluster.
8. CLIP cosine cross-check; LPIPS on top-K suspicious gen/train pairs.
9. Save suspicious-sample metadata for panels.

---

## 5. Outputs & schemas

- `results/experiment4/per_seed_results.csv` — columns: `regime, checkpoint_type, seed, checkpoint_path, sample_count, outer_step, effective_feu, val_fid, val_kid, train_fid, train_kid, fid_gap_val_minus_train, kid_gap_val_minus_train, mean_nn_train_dist, mean_nn_val_dist, nn_train_val_ratio, nn_frac_ratio_lt_0_9, duplicate_rate, near_duplicate_rate, status, n_seeds_in_arm`.
- `results/experiment4/aggregate_summary.csv` — rows {step,compute}×{vanilla,anm,paired_delta}; cols mean/std/sem/median/95%-bootstrap-CI for val_fid, val_kid, fid_gap, kid_gap, nn_frac_ratio_lt_0_9, dup_rate. Include `n_seeds` and a `seed_audit_valid` boolean (true only if n_seeds≥3 per arm).
- `results/experiment4/nn_stats.json` — full distributions + suspicious indices.
- `results/experiment4/nn_panels/` — `top_32_lowest_train_distance`, `top_32_lowest_ratio`, `top_16_largest_margin`, `top_duplicate_clusters`, `random_16_control`; layout gen | top-3 train NN | top-3 val NN; `nn_panels_metadata.json`.
- `plots/` — `seed_error_bars_fid.png`, `seed_error_bars_kid.png`, `train_val_gap_fid.png`, `train_val_gap_kid.png`, `nn_ratio_distribution.png`, `duplicate_rate_by_seed.png`, `step_vs_compute_summary.png`. Show individual seed points; pair vanilla↔anm per seed with lines; label whether error bars are std or 95% CI.

---

## 6. Controls (assert at runtime; fail loud)

- No val image in training or in ANM mining; train/val bank file-IDs disjoint (`assert`).
- Identical fixed eval seeds + label schedule + sampler + NFE + stepsize + CFG + EMA + VAE + sample count for every checkpoint.
- One frozen Inception for FID/KID; cache features with metadata (backbone, hash, preprocessing, resolution, split, subset seed).
- Equal train-ref and val-ref counts; identical class balance.
- Rule-based ckpt selection only (no best-by-val-FID, no best-seed-after-eval).
- ANM inner rollout `no_grad`/eval; `δ` stop-gradient; mining freq/K logged; counter persisted.
- Reference-gap calibration: also compute `FID(train_ref_A, train_ref_B)` and `FID(val_ref, train_ref_B)` to separate model overfit from natural split mismatch.
- NN sanity: val-vs-train distances plausible; inserted exact duplicate detected; random noise not suspiciously close; generated samples not accidentally in the banks.

## 7. Failure detection (explicit error, never silent)
missing checkpoint · missing reference features · wrong train/val split (path/ID check) · generated count mismatch · inconsistent eval seeds (hash the schedule, store it, compare) · feature-extractor preprocessing mismatch (store preprocessing hash in the cache) · NN bank too large for memory (tile + running top-k) · train/val leakage (disjoint-ID assert) · compute-matched ckpt absent (`BLOCKED_NEED_CKPT` row + exact train command).

---

## 8. Smoke command (build first, on cluster GPU)
```
# tiny: N=1000, ref=5K, nn bank=10K, 1 vanilla + 1 anm seed0
bash scripts/cluster/remote_submit.sh projects/diff-EqM/slurm/jobs/exp4_smoke.sbatch diff-EqM
```
Pass criteria: config parses; samples generate (or load); features extract; FID/KID finite; both references scored; NN returns valid IDs; inserted duplicate detected; CSVs + JSON + panels + plots written; rerun reproduces features within tolerance.

## 9. Full-run commands
```
# 0. references (once): val + class-balanced train subset, unified {mu,sigma} schema
bash scripts/cluster/remote_submit.sh projects/diff-EqM/slurm/jobs/exp4_build_refs.sbatch diff-EqM
# 1. step-matched
bash scripts/cluster/remote_submit.sh projects/diff-EqM/slurm/jobs/exp4_step_matched.sbatch diff-EqM
# 2. compute-matched (launch extended-vanilla train first if ANM-final budget ckpt missing)
bash scripts/cluster/remote_submit.sh projects/diff-EqM/slurm/jobs/exp4_compute_matched.sbatch diff-EqM
```
Each sbatch follows the repo template (clone→/tmp, checkout `GIT_SHA`, pip incl. dinov2/open_clip/lpips, torchrun, rsync `results/experiment4` + `plots` back). Mining variants/NN search → `gpu`/`seas_gpu` (full A100), not `gpu_test` 20G (OOM).

## 10. Interpretation notes (report verbatim categories)
- **Clean support:** lower mean val FID/KID, std not worse, paired wins in most seeds, no widened gap, no higher train-NN proximity, no higher dup rate, benign panels.
- **Step win / compute loss:** ANM helps per step but gain may be compute → not a compute-efficient generalization claim.
- **Val win + widened gap:** suspicious; require NN + panels before accepting.
- **Lower train FID, flat val FID:** overfit/bias, not the ANM story.
- **Better mean, higher variance:** unstable; report per-seed, do not average away.
- **One-seed win only:** **failure** for stability (this is the current B2 state → variability audit, not seed audit).
- **Higher dup/near-dup:** veto even if FID improves.
- **Panels show copying:** memorization risk; report gen ID + train NN ID + distance + class + ckpt meta.
- Final verdict must be one of: **Pass / Mixed / Fail**, with the n_seeds caveat attached whenever `seed_audit_valid=false`.

---

## Required final output from the coding agent
1. Corrected discovered file/function map. 2. Exact implementation plan. 3. Files to add/modify. 4. Complete code or diff. 5. Example checkpoint config. 6. Smoke command. 7. Full-run commands. 8. Output schemas. 9. Plotting plan. 10. Interpretation notes. **Plus** explicit statement of compute-matched checkpoint availability and the B1/B2 caveats.
