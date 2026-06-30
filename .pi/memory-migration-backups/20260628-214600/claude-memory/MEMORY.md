# Memory for research-repo Project

## RC-HPM (ACTIVE)
- [rc-hpm CPU tree complete](rc_hpm_cpu_tree.md) — machinery validated (G-1/G0); harm-bounding framing (G1); EqM bridge dead at toy via oracle-null (G2); Stage-2 decision pending user.

## EqM Metacognition / Separability (GREEN — reopened via dynamics probe)
- [EqM separability diagnostic](eqm_separability_diagnostic.md) — energy scalars DEAD (dot 0.609/path 0.605 < dumb baseline 0.627) BUT learned probe over descent-SHAPE (oscillation+normalized curves) = 0.813±0.002 de-confd (5 seeds), crosses 0.80. Good/garbage signal in descent DYNAMICS not energy. Metacognition sampler REOPENS. Next: held-out + wire probe into sampler + FID test. `learned_probe.py`, FINDINGS.md.

## Symmetry Discovery Pivot (ACTIVE)
- [v16 conclusions WRONG — distrust](symmetry_v16_distrust.md) — v14-beatcrop/v15/v16 negatives invalid per user (2026-06-06); symmetry-discovery route CONTINUES w/ pivot. Don't cite "bridge concluded / nothing to discover / no headroom over crop".
- [v17 MorphismGym — FIRST POSITIVE](symmetry_v17_morphismgym.md) — hidden-morphism gym (valid on-manifold, decoys off; label-free PCA-whitened anchor AUC 1.0). Phase 0-3 ALL PASS + dSprites confirm: discovery BEATS random + approaches/beats oracle on EqM-lite across 3 image distributions (SDF shapes/MNIST/dSprites, 3 seeds each), anchor essential. First discovery>random in v13-v17; targeting-without-labels SOLVABLE w/ real hidden structure. Write-up `documentation/v17_morphismgym_writeup.md`. Bridge port SPEC `documentation/v17_to_eqm_bridge_spec.md` (MULTI-family+whitened-anchor+bandit supersedes old single-operator v12/v13); cluster session LIVE but bridge NOT submitted — blocked on human FID-approval + diff-EqM greenlight (FID never auto). Caveats: synthetic/MNIST/dSprites controlled-latent + EqM-LITE proxy, multi=single on payoff.

## Analysis & Documentation Rule

**CRITICAL**: When conducting analysis of experiments/results, ALWAYS update ALL relevant state files:
- `pipeline.json` - phase transitions, analysis summaries, next actions
- `documentation/evaluation-analysis.md` - comprehensive findings
- `documentation/debugging.md` - issues, root causes, failed experiments
- `documentation/research-plan.md` - next steps, hypotheses, decision points
- `documentation/queue.md` - action items, priorities, blockers

**Principle**: State files are the project memory. Every analysis, finding, and decision must be logged so that future work can build on complete context without re-discovering knowledge.

## QOS Management & Partition Diversification

**Key Insight**: When submitting 4+ jobs in a single dispatch operation, the cluster's per-partition QOS limits will block submissions if all jobs target the same partition.

**Solution**: Distribute jobs across `gpu_test` and `gpu` partitions
- Split batch roughly evenly: ~N/2 to gpu_test, remaining to gpu
- Allows simultaneous job submission without hitting "QOSMaxSubmitJobPerUserLimit"
- Jobs run in parallel without interference across partitions

**Applied in algebra-ebm**: Submitted 6 evaluation jobs (3 to gpu_test, 3 to gpu) successfully

**Updated in CLAUDE.md**: Added "QOS management & partition diversification rule" section for future use

## Project Phase Tracking

### Algebra-EBM Project (projects/algebra-ebm)
- **Completed**: Full evaluation suite (6 experiments) after DATAGEN-001 through 005 fixes and model retraining
- **Current Phase**: DEBUG (inference diagnostics and investigation)
- **Critical Finding**: Compositional EBM approach fundamentally broken - training works (9-10 unit energy gaps) but inference fails catastrophically (6% single-rule, 0% multi-rule). Root cause: IRED gradient descent stuck in local minima.
- **Key Metrics**: Single-rule expected 85%, observed 6.3% avg. Multi-rule expected 30-70%, observed 0%.
- **Next Steps**: Inference diagnostics experiments (multi-start, increased iterations, momentum, better initialization)

### IRED Project (projects/ired)
- **Completed**: q211 baseline (8 seeds), q220 TAM baseline (8 seeds), q221/q222 hyperparameter sweeps, **checkpoint infrastructure fixed**, q223_test_minimal training
- **Current Phase**: FULL TAM-CTL TRAINING (q225) → OOD ROBUSTNESS EVALUATION (q226)
- **NOW RUNNING**: q225 Full TAM-CTL Training
  - Job ID: 61961915 (8 seeds, array 0-7%2)
  - Status: PENDING → RUNNING on gpu partition
  - Submitted: 2026-02-24T16:15:00Z
  - Expected Runtime: ~2h per seed, ~4h wall-clock
  - Expected Completion: 2026-02-24T20:15:00Z
  - Configuration: TAM mining + recovery loss (λ=0.1), hyperparameters optimized from q221-q222
  - Purpose: Generate checkpoint for q226 OOD robustness evaluation
  - Git SHA: 17db7ca (unbuffered output + tee for better debugging)

- **Recent Milestones**:
  - q223_test_minimal: 100-step SUCCESS (job 61945057, exit code 0)
    - final_model.pt (5.5M) successfully saved and synced to persistent storage
    - Checkpoint loading verified, recovery loss working correctly
  - q221: TAM anchor_step sweep - anchor_step=2 OPTIMAL (0.009731 ± 0.000011)
  - q222: TAM pgd_delta sweep - NEGLIGIBLE sensitivity across deltas 0.5-3.0
  - Both sweeps validate q220 baseline hyperparameters are well-chosen

- **Next**:
  1. Early poll at 30min to catch initialization errors
  2. Wait for q225 completion (~4h from submission)
  3. Retrieve results and compare q225 vs q220 (measure recovery loss impact)
  4. Save q225 checkpoints for q226 OOD robustness evaluation
  5. Proceed with q226: OOD matrix evaluation using q225 checkpoint

## Experiment Implementation Pattern: Build Incrementally, Not in Grid

**CRITICAL LESSON from q220 TAM failure:**

**What FAILED (q220):**
- Implemented TAM core logic
- Immediately jumped to 45-config grid sweep (delta × pgd_steps × anchor_step × ctl_weight × gc)
- Submitted all 45 in parallel with --array=0-44
- Result: Core TAM logic untested, impossible to know which (if any) variants work

**What WORKS (q216-q218 success pattern):**
1. **q216**: Test 1 variant in isolation (Gradient-Contrastive alone)
   - Collect results
   - Analyze before moving on
2. **q217**: Test 1 new variant in isolation (PGD adversarial alone)
   - Collect results
   - Analyze before moving on
3. **q218**: Test 1 variant of previous (PGD centered on positive)
   - Collect results
   - Analyze and decide next step

**Correct TAM approach (DO THIS):**
1. **q220 (BASE)**: Single TAM config with good defaults
   - Validate core TAM logic even works at all
   - If it works, do Step 2; if broken, debug and iterate on q220
2. **q221 (SWEEP)**: Only after q220 validates TAM → mini-sweep of 45 variants
   - Now you have a known-good foundation
   - Sweep results are interpretable

**Rule**: Never submit multi-dimensional grid sweep without first validating 1-dimensional core logic works.

## ImageNet Data Blocker
- [ImageNet not on cluster](imagenet_blocker.md) — empty dirs, blocks EqM-B/2 training. Pipeline ready, just needs data.

## Cluster Storage (load-bearing)
- [IN-1K ckpts → holylabs, not home](diff_eqm_holylabs_quota.md) — home03 95G cap; ckpts-to-home deadlocks trains (RUNNING but frozen, 12h gate loss 2026-06-04). Set PERSISTENT_RESULTS=holylabs; verify liveness via stdout/ckpt mtime.

## diff-EqM Variant Findings
- [v10 IN-1K 3-seed win](diff_eqm_v10_in1k_3seed.md) — v10 B/2 80ep FID 27.58±0.36 vs vanilla 31.41 (−3.83), all 3 seeds below. Phase1 PASS. Phase2 Welch blocked on vanilla seeds1,2 (failed sync, ep57-59). FID sbatch needs explicit GIT_SHA (HEAD fails on fresh clone).
- [DG-ANM variant search results](diff_eqm_variant_findings.md) — v02 beats vanilla 12.96±0.70 vs 14.17 on variant harness (R4 150ep). NEVER compare FIDs across variant_harness vs legacy cifar_seed_study runner (4.7 FID harness gap).
- [Stage B vanilla baseline verified](diff_eqm_baseline_verified.md) — IN-1K-256 EqM-B/2 80ep FID=31.41 (paper 32.85, Δ=-1.44). Pipeline reproduces paper; v10 work unblocked.
- [Branch B-Both framing locked](diff_eqm_framing_branch_b_both.md) — first adaptive hard-negative mining for regression-target generative models, PGD-on-EqM-target (v10) × CAFM (Lin 2026) discriminator. NeurIPS 2026 workshop + ICLR 2027. Two external future-work cites (VeCoR §7 + EqM paper) support direction.
- [Phase 0.3 v10 PASS](diff_eqm_phase_0_v10_pass.md) — v10 CIFAR 150ep FID 13.40 beats vanilla 14.17 by 0.77; mining ratio 1.047-1.049 stable (non-saturating, the key differentiation from v02). Unblocks Phase 1a CAFM-EqM run.
- [Exp 2 field robustness](diff_eqm_exp2_field_robustness.md) — ANM off-trajectory field-robustness mechanism CONFIRMED at IN-1K B/2 (lower MSE/higher cos all radii SIG, largest at real v10-mined δ dMSE −0.0368) but effect small. Includes reusable cluster gotchas (--export comma split, no_grad-not-inference_mode, transport vs CAFM target sign).
- [Exp 3 fidelity-diversity](diff_eqm_exp3_fidelity_diversity.md) — STRONG_SUCCESS no diversity tax: ANM l03 FID 26.88 vs van 31.27 (−4.38 disjoint CIs), recall FLAT, coverage +0.072, weak-class bottomQ −5.61, 91% classes improve (single seed). Eval-infra gotchas: holylabs GROUP quota EDQUOT invisible to df; split read/write fs; extractor-skip feats/stems align bug; score on common id set.
- [Symmetry vs mining toy ladder](diff_eqm_symmetry_ladder.md) — CONCLUDED (rungs 1-11), own project `projects/symmetry-discovery/`. Mining (v10) installs NO manifold structure; KNOWN-symmetry = clean. Unsupervised discovery VIABLE/NEAR-ORACLE with a CLASS-AGNOSTIC prior (rungs 12-14): frozen data anchor + coherent op + group generator exp(A) + STABILITY-ONLY reg (det~1, cond->1). NO rotation/plane/class naming -- symmetry+active subspace EMERGE. Rung14 (K=3, stable!=rotation) = oracle. SCOPE: validated only for ROTATION/ISOMETRIC symmetry in a CLEAN aligned latent. EqM BRIDGE built (v12 variant in dganm_variants/ + feature proxy) but mechanism FAILS on real CIFAR: fast feature-space proxy (frozen random-conv+PCA, rotation gap) shows discovered stable-generator is stable+beats-random but ~=BASE on held-out gap (anchor-matching doesn't target held-out; feature space not symmetry-aligned). KNOWN_AUG closes gap (works). Do NOT extend CIFAR/FID for v12. CIFAR/FID 40ep noise-limited (v00 228,v10 205,vK 219,v12rand 218). Fast proxies (feature_gap_proxy_*) decompose the failure: flat-PCA dense-M = wrong ARCHITECTURE (~=base); SPATIAL affine op M=exp(A) via grid_sample learns a CLEAN coherent rotation (det/cond 1.00) = architecture SOLVED. Remaining open problem = TARGETING: anchor-distribution-matching is direction-agnostic, learns wrong-direction rotation vs a ONE-SIDED held-out band -> gap worse; residual-anchor also fails. Toy rungs 12-14 worked only b/c held-out was SYMMETRIC under the group. KNOWN-aug works (gate passes). Verdict: operator architecture done; held-out targeting without labels is the unsolved bottleneck. Recommend known-symmetry aug for paper; targeting-objective = research extension. Don't scale FID until targeting solved. CRITICAL: held-out recall is COVERAGE-confounded — rung15 (learned latent) + rung16 (screw) get high recall via on-manifold reshuffle WITHOUT coherent symmetry; trust operator-quality metrics not recall. rung15 learned-latent=partial (incoherent). rung16 screw=INCONCLUSIVE x2 (ill-posed). Ladder PAUSED. EqM bridge plan written (eqm-bridge-plan.md: v12 frozen stable-generator aug vs frozen feature anchor, CIFAR, compare v00/v10/known-aug). NEXT HUMAN ACTION: `! scripts/cluster/ssh_bootstrap.sh` (2FA) for bridge, or pick toy sub-problem (aligned latent / better non-rotation testbed). Prefer symmetry constraints over more mining.

## Critical Bug Fixes

### q211 Baseline DataLoader Freeze (FIXED - commit 98521e8)

**Problem**: q211 baseline (job 61637386) seeds 0-6 all FAILED mid-training at steps 3.3K-4.9K
- Seed 0: stopped at step 3500
- Seed 1: stopped at step 3300
- Seed 2: stopped during evaluation
- Seed 3: stopped at step 4900
- Seed 7: also stopped during evaluation (later in queue, saw same issue)
- Seeds 4,5,6: never started (throttled by %4 queue policy)

**Root Cause**: `denoising_diffusion_pytorch_1d.py:1375` used `self.data_workers = cpu_count()`
- Cluster node has 64 CPUs → spawned 64 DataLoader workers
- PyTorch warning: "Our suggested max number of worker in current system is 2"
- 64 workers with batch_size=2048 causes massive memory contention → process freezes
- SLURM kills frozen process, logs show incomplete training

**Fix Applied** (commit 98521e8): Changed line 1375 to `self.data_workers = 2`
- Follows PyTorch best practices for DataLoader configuration
- Prevents memory contention and process hanging
- Matches system-recommended worker count

**Next Action**: Cancel q211 job 61637386, resubmit with fixed code (git SHA will be 98521e8 or later)

### TAM-CTL Recovery Config Bug (FIXED - commit 4e30151)

**Problem**: q223 TAM-CTL (jobs 61758176, 61761168) running without recovery loss enabled despite config having use_recovery_loss: true
- Logs showed only CD-DIAG (energy-based loss) output, NO TAM-CTL-DIAG
- Diagnostics indicated recovery mechanism wasn't executing
- Root cause: recovery loss params never passed to GaussianDiffusion1D

**Root Cause**: In `projects/ired/experiments/matrix_inversion_mining.py:104-154`:
- mining_config dict was being extracted from config JSON
- BUT it was missing 3 critical fields:
  - `use_recovery_loss`
  - `recovery_steps`
  - `recovery_loss_weight`
- Even though config file had `"use_recovery_loss": true, "recovery_steps": 1`, they were never added to the mining_config dict
- Result: Line 1019 in denoising_diffusion_pytorch_1d.py would always get recovery_steps=0, so recovery was disabled

**Fix Applied** (commit 4e30151): Added extraction of recovery params in experiment.py:
```python
# TAM-CTL (Convergence Training Loss): recovery objective
'use_recovery_loss': config.get('use_recovery_loss', False),
'recovery_steps': config.get('recovery_steps', 1),
'recovery_loss_weight': config.get('recovery_loss_weight', 0.1),
```

**Impact**:
- Previous jobs 61758176 and 61761168 cancelled
- Resubmitted as job 61767772 with fixed code (commit 4e30151)
- This is a critical lesson: config→mining_config extraction must be exhaustive or silent failures occur

**Key Lesson**: When passing config dicts to different modules, explicitly verify that ALL used fields are extracted. Silent config field omissions are harder to debug than explicit errors.

### TAM-CTL Recovery Loss Shape Bug (FIXED - commit 5f31dbd)

**Problem**: Job 61767772 failed at initialization with RuntimeError in recovery loss computation
- Error: "The size of tensor a (400) must match the size of tensor b (2048) at non-singleton dimension 1"
- Line 1193: `loss_rec = F.mse_loss(pred_rec, target, reduction='none')`

**Root Cause**: Using `target` variable in recovery loss computation, but `target` didn't have the correct shape
- At line 859, `target = noise` for pred_noise objective
- But using explicit variable name `target` is confusing and error-prone
- The variable `noise` is the explicit noise tensor with correct shape [B, seq_len]

**Fix Applied** (commit 5f31dbd): Changed line 1193 to use `noise` instead of `target`
- `loss_rec = F.mse_loss(pred_rec, noise, reduction='none')`
- Clearer intent: matching pred(y_rec) to the noise used for the diffusion forward process
- Avoids variable aliasing confusion

**Impact**: Job 61768617 submitted with this fix plus the config extraction fix
- This is 4th attempt at q223 (61758176, 61761168, 61767772, 61768617)

**Key Lesson**: Be explicit about variable names and avoid aliasing (target = noise). Use the direct variable name that reflects intent.

### Checkpoint Persistence: Config Path Bug (FIXED - commit 3024170)

**Problem**: Checkpoint infrastructure appeared broken - training said files saved successfully, but rsync didn't sync them and debug showed no .pt files in work directory.

**Root Cause Investigation**:
1. Training script runs: `cd "$WORK_DIR"` then `python projects/ired/experiments/matrix_inversion_mining.py --config projects/ired/configs/q223_test_minimal.json`
2. Config has relative path: `"output_dir": "results/ds_inverse/q223_test_minimal"`
3. Training interprets this relative to pwd = `$WORK_DIR`, saves to: `$WORK_DIR/results/ds_inverse/q223_test_minimal/`
4. But rsync syncs from: `$WORK_DIR/projects/ired/results/`
5. **Result**: TWO DIFFERENT DIRECTORIES — training outputs never matched rsync source!

**Fix Applied** (commit 3024170): Updated all 37 config files from:
```json
"output_dir": "results/ds_inverse/q223_test_minimal"
```
To:
```json
"output_dir": "projects/ired/results/ds_inverse/q223_test_minimal"
```

**Validation** (job 61945057):
- Trained 100 steps with fixed config
- Checkpoint appeared in work directory: `/tmp/ired-job-61945057/projects/ired/results/ds_inverse/q223_test_minimal/final_model.pt` (5.5M)
- Rsync successfully synced to persistent storage
- ✓✓✓ SUCCESS message in logs

**Key Lesson**: When scripts change directory before running training, relative paths in config are relative to pwd, not to script location. Always use absolute or explicit paths that match where downstream operations expect files.
- [Capability ladder NULL](diff_eqm_capability_ladder_null.md) — v10 FID gain does NOT transfer to image-repair (inpaint/denoise/colorize/superres/deblur); Rung1+2 v10~vanilla noise-level. Ladder killed per gate; control train + Rungs3-5 not run. v10 = generation method only.
- [Capability ladder v2 (A-F) — gain bounded near-manifold + ANM-vs-symmetry BENCHMARK](diff_eqm_capability_ladder_v2.md) — v10 FID gain IS real+behavioral: A/B/D positive (quality+hard-class 1.34-1.9x+2.5x sample-eff, no collapse), C/E/F null (no rescue/steer/splice-loc). Mining sharpens field NEAR manifold only. SUPERSEDES v1 inpainting-null pessimism. **A-F = SHARED benchmark for ANM vs symmetry-discovery: v10=baseline bar, run symmetry ckpt same protocol, test beat-on-A/B/D or light-up-null-C/E/F.** Ops: sbatch --export comma-split bug, home03 exit-53 quota, run pruner.
- [v20 metacog-mine CIFAR PASS](eqm_separability_diagnostic.md) — instability-mining (label-free, ||Δf|| selector) FID 13.26 < random/vanilla 14.41 (-1.15, gain from failure-SELECTION; random==vanilla 4dp). Metacog works as training selector on IMAGES (Sudoku probe-selector was weak). ~=v10 PGD cheaper. IN-1K confirm next (human-gated FID).
