# EqM Separability Diagnostic

**Question (the whole point):** does *any* cheap scalar, computable at the moment
sampling "goes quiet" (gradient norm small), reliably assign worse values to
**garbage** outputs than to **good** ones — on the **vanilla EqM-B/2 (80ep)**
checkpoint, no training?

This gates two possible downstream projects (a metacognitive adaptive sampler;
a conservativity fix to make the scalar energy reliable). It is a **measurement**,
not a model change. A **KILL** verdict is a fully successful experiment — it saves
weeks. Do not mistake a null result for a bug (see "success of script vs hypothesis").

## The 2×2 it tests
Two signals at the stopping point: gradient norm `‖f‖` (confidence about the next
move) × energy level (am I on the manifold?). The load-bearing cell is
**low norm + high energy = spurious minimum** ("thinks it's done, on garbage") —
invisible to a norm-only stopping rule. We need an energy axis that is
*independent of the norm* to detect it. This experiment checks whether one exists.

## Pipeline (run in order; the sbatch chains all four)
1. `sample_with_logging.py` — fork of the FID-trusted GD sampler; logs per step
   `norm, dot=<f,x>, l2=0.5‖f‖², step_dot=<f,dx>`; decodes PNGs; writes per-rank
   `logs/traj_rank*.npz`. **Sign convention `xt = xt + f·η` is verified — do not flip it.**
2. `compute_quality_labels.py` — **independent** good/garbage label = Inception
   (pool3, FID feature space) k-NN distance to a bank of real ImageNet images;
   thresholds fixed from the distribution (good = bottom 40%, garbage = top 30%,
   drop middle). resnet50 max-softmax as a direction cross-check. Also VAE-encodes
   the real images → `real_latents.npz` for s4.
3. `compute_scores.py` — 5 candidate scalars at the stopping point, two regimes:
   `threshold` (first `‖f‖<τ`, τ∈{5,10,20}) and `fixed` (last step = de-confound).
   - `s1=-dot`, `s3=path-integral` → **de-confoundable** (can carry norm-independent info)
   - `s2=0.5‖f‖²`, `s5=post-step ‖f‖` → **norm-coupled** by construction
   - `s4=latent-NN dist` → no `f`; a label-sanity positive control
4. `analyze.py` — raw AUROC + **within-norm-bin AUROC** (the matched-norm control)
   per score/regime; histograms; `results/VERDICT.txt`.

## Verdict logic
`best_independent_auroc = max(within-norm-bin AUROC of s1, s3)`:
- **≥ 0.80 → GREEN**: norm-independent energy signal exists; spurious-minimum
  quadrant detectable; build the metacognition sampler next (now de-confounded).
- **< 0.60 → KILL**: no local signal separates true vs spurious minima
  independent of norm; both downstream ideas high-risk. If s2/s5 *do* separate,
  the matrix collapses to one row (norm) — explicitly noted.
- **0.60–0.75 → WEAK**: separation exists but sub-threshold; output quantifies how
  much a "more reliable scalar" must improve.

## Run
```bash
# smoke first (200 samples, ~10 min on 1 GPU)
GIT_SHA=<sha> CKPT_PATH=projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0 \
  NUM_SAMPLES=200 NUM_REAL=2000 sbatch projects/diff-EqM/slurm/jobs/sep_diag.sbatch
# full
GIT_SHA=<sha> CKPT_PATH=projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0 \
  NUM_SAMPLES=3000 NUM_REAL=10000 sbatch projects/diff-EqM/slurm/jobs/sep_diag.sbatch
```
`CKPT_PATH` may be the dir (highest nested `.pt` auto-picked) or the exact file
`.../000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt`.

## Sanity checks (analyze + logs surface these)
1. Decoded "good" tail looks like real classes; "garbage" tail looks like mush.
2. `norm` **decreases** over the trajectory on average (else a sign bug in the fork).
3. `s2` AUROC ≈ a norm-threshold classifier (pipeline wired correctly).
4. `s4` (no `f`) raw AUROC moderate-to-high; if ≈0.5 the labels are broken.
5. Smoke (200) end-to-end before the full 3000.

## Script success vs hypothesis success
Script success: all 4 stages run, `VERDICT.txt` produced, sanity checks pass.
Hypothesis success is the *separate* question the verdict answers — and **KILL is
a successful experiment**, not a failure of the script.
