---
name: eqm-separability-diagnostic
description: EqM-B/2 metacognition signal REOPENED — descent-SHAPE (not energy) separates good/garbage 0.81 de-confounded
metadata: 
  node_type: memory
  type: project
  originSessionId: 4c4d13ce-a589-4e84-9f2f-d07d5534b743
---

**GREEN UPDATE:** energy scalars stay dead, BUT a LEARNED probe over the descent-
trajectory SHAPE (oscillation + magnitude-normalized norm/dot curves) hits
within-norm AUROC **0.813±0.002 (5 seeds)** — crosses 0.80, beats norm floor
(0.684) + dumb baseline (0.627). Good/garbage signal lives in descent DYNAMICS,
recoverable by a small probe, NOT any single scalar/energy. Metacognition sampler
REOPENS via learned dynamics probe. Stage 5 `learned_probe.py` (CPU numpy LR ~1s).
**PAYOFF CONFIRMED (end-to-end):** held-out 0.818±0.012 (30% test, 5 seeds) +
probe-guided rejection beats random floor at EVERY keep-frac (0.5-0.9), recovering
34-50% of inception-oracle FID gain (e.g. keep0.7: probe 86.7 vs random 100.2±0.3
vs oracle 73.1). Trajectory-shape probe (no image access) actionably improves
generation. Scripts: learned_probe.py / probe_validate.py (saves probe_artifact.npz)
/ fid_payoff.py (+ fid_payoff.sbatch). PROXY FID (3000 subset,10k ref,cfg1) — only
relative ordering valid.

**JUN23 PARETO-NULL + ONLINE LOCKDOWN (50k, FID-gated reads):** PI airtight-result asks
ANSWERED. Exp1 (restart vs depth, NFE=750 matched, jobs 24249662-679): long250=28.17,
long750(depth)=29.49 WORSE than long250 -> deeper vanilla GD saturates, adds nothing;
r3rand=28.03, r3energy(Σ‖f‖)=25.78, **r3probe@50=24.84 wins** (+4.65 vs depth, +3.19 rand,
+0.94 best-trivial). NULL KILLED: restart>depth, early-probe=best selector. Preview 8k agrees
(long750 33.23 vs r3probe 28.67). Exp2 online early-abandon @step50 (job 24249680): vanilla
27.93/random_restart 27.89/**probe_restart 26.46(+1.43)**/oracle 21.75 -> recovers 23% oracle
gain reallocating compute online at equal NFE. SELECTOR_LOCKDOWN_RESULTS.md updated w/ both.
Lockdown 2-seed: probe@50 24.84/24.65 < energy_path 25.78/25.75 < vanilla 28.2/27.8; FULL-traj
probe does NOT cleanly beat energy_path(Σ‖f‖)=trivial -> sharpened claim = EARLY probe wins,
edge grows earlier you read. OPEN: 5-seed CI needs trimmed-arm rerun (3/5 seeds OOM'd 1-rank).

**JUN18-19 OVERNIGHT SCOPE-MAP (3 threads, PUSHED 2b33929):** User AFK, asked: does it
generalize better than vanilla / improve other tasks / can inpaint be made positive. Built
NIGHT_PLAN. RESULTS:
(1) THREAD1 OOD-GENERALIZATION = WIN: maze frozen in-dist(c7) probe across c5-c13 OOD, fixed
NFE. probe-vanilla advantage GROWS MONOTONE w/ shift **corr 0.98** (+0.078->+0.182), frozen
probe TRANSFERS to +6 OOD (AUROC~0.8 all tiers), probe>random every tier. s0 near-ceiling
control=little gain (need failures to rescue). maze_ood_scaling.py, OOD_SCALING_RESULTS.md.
(2) THREAD2 SUDOKU-EqM (3rd task=CSP/constraint-reasoning): DETECTION transfers, ACTION
doesn't. 9x9 too-hard (loss 2.4->0.99 but solve 0/all-81-cell, gate FAIL - small GD-EqM
can't do full grid, NOT metacog). 4x4 gate PASS solve 0.955. 4x4 metacog: probe DETECTS
constraint-violation AUROC **0.84-0.90** de-conf (signal transfers to CSP!) BUT restart gap~0
b/c oracle~=random -> R=4 restarts CORRELATED (deterministic/puzzle-intrinsic failures, no
lucky restart). KEY: detection!=actionability; best-of-R needs STOCHASTIC failure (gen/maze
have it, CSP doesn't). sudoku_eqm.py/sudoku_metacog.py (grid-param N, synthetic gen),
SUDOKU_RESULTS.md. ired NOT present locally (no sudoku to reuse).
(3) THREAD3 INPAINT POSITIVE REGIME FOUND: dual oracle - CLASSIFIER(semantic/confident-wrong)
stays CHANCE 0.56-0.61 ALL masks; STRUCTURAL(instability, label-free connected-components) at
extreme mask 0.90 -> AUROC **0.84** +restart **0.18** (random-mask AUROC-vs-mask corr 0.88
monotone). Inpaint becomes positive EXACTLY when extreme masking shifts failures confident-
wrong->structural-collapse(instability). CONFIRMS scope law (Jun15 null + this positive = same
mechanism at 2 mask sizes). mnist_difficulty_sweep.py, INPAINT_POSITIVE_RESULT.md.
NET LAW: signal=INSTABILITY in relaxation; transfers across tasks for DETECTION; restart-ACTION
pays where failures are (a)instability-type AND (b)stochastic. SYNTHESIS links #6-8 + scope
section now a TESTED law. SSH 2FA-dropped ~23:00 (needs bootstrap, user asleep) -> 4x4+inpaint
ran LOCAL CPU (no GPU needed); no jobs lost. OPEN: 9x9-harder/bigger-arch, image-RePaint, CSP
non-restart action (extra-compute-on-flagged), all SSH-gated.

**JUN18 CLUSTER-RETURN RESULTS (PUSHED bd0465e):** MGHPCC back. Fired queued jobs.
(1) ONLINE 50k 3-draw = WIN/decision-grade: probe-restart FID **26.90 < random {27.89,27.96,
28.01} ALL draws** (Δ~1.05 equal-NFE), oracle 21.75, vanilla 27.93. Promotes online claim
15k->50k. HONEST: only random-restart arm reseeds (probe/vanilla/oracle deterministic given
ckpt) -> "probe robustly beats random control", NOT 3 indep probe seeds (the 50k best-of-R
Δ1.87±0.11 covers genuine-3-seed). SCALE_RESULTS.md + SYNTHESIS row4.
(2) MAZE GPU scale-up CONFIRMS+TIGHTENS but TRAIN UNSTABLE: width128/80ep across 3 seeds =
s0 in-dist0.975(great)/s1 0.68(ok)/s2 DIVERGED loss20 in-dist0.0 gate-FAIL (recipe issue:
LR-too-high/no-warmup, EXCLUDED). Metacog @proper budget: **s1 in-band AUROC 0.872 Δ+0.175
87%oracle = BEST maze yet** (>CPU 0.67-0.76, 3.7x wider GPU EqM); s0 near-ceiling(van0.76)
+0.022. Both probe>random. Corroborates CPU multi-seed +0.174 (NOT replaces; 1 in-band seed).
BUG root-caused: 1st metacog run failed (vanilla c10 ~0) NOT model/probe -> sampler budget
0.25 (eta0.01 hardcoded in maze_gpu.sbatch) BELOW manifold threshold; fixed eta>=0.02 via new
maze_sampler_probe.py. Also fixed maze_metacog width-mismatch (infer C from ckpt args, was
hardcoded C=64 vs C=128 ckpt) + gd_sample log=False arity. STEP3_GPU_RESULTS.md, ladder D".
OPEN/optional (no gate forces): width128 recipe-fix for clean 3-seed GPU maze; RePaint-IN1K.

**MNIST INPAINTING RUNG = WEAK/NULL + SCOPE BOUNDARY (2026-06-15, PUSHED caa33db,
`experiments/mnist_eqm/`):** Reused real MNIST + RePaint-clamp masks + classifier-
consistency oracle + small EqM (loss 2.66, clf acc 0.967). Metacognition 3 masks x 3 seeds:
probe AUROC ~0.60 (~chance), probe-restart gap +0.015±0.011 pooled (7/9 pos) — effectively
NULL, far below maze (0.67-0.76, +0.17). Full & masked dynamics both fail (full=-0.045
center0.55). KEY INSIGHT (sharpens paper not weakens): metacognition keys on descent
INSTABILITY -> rescues collapse/broken-structure failures (generation, maze planning) but
NOT confident-wrong failures (inpaint descends CLEANLY to plausible-wrong digit, masked 4->9;
dynamics healthy, identity wrong, probe can't grip). Failure-mode-specific. RePaint-on-IN1K
image inpaint spec'd for Jun-18 (predicted weak same reason). SYNTHESIS_METACOGNITION.md =
6-link cross-task narrative + scope boundary. Maze-inpaint abandoned (too easy: clamp+given-
walls=100% valid). CLUSTER DOWN Jun15-18 (MGHPCC power, NOT VPN) -> maze-GPU+online-50k
queued in scripts/cluster/fire_overnight.sh, auto-fire on return. SCOOP: CFG-Rejection
(2505.23343) concurrent but ASD needs CFG (undefined for EqM cfg=1.0), post-hoc/images/no-
deconfound -> differentiated.

**REAL EqM MAZE PLANNING (2026-06-14, PUSHED d77da51, `experiments/maze_eqm/`):**
Yilun's maze suggestion done on a GENUINE trained EqM (not toy/not image-ckpt). 3 steps:
(1) data: grid-maze->shortest-path, perfect-maze DFS + BFS validity ORACLE (exact, no
inception noise), tiers c5/c7/c10 (11/15/21 grid). (2) trained small conditional EqM
(653K UNet, FAITHFUL EqM target c(t)=min(1,5(1-t))*4 replicated from transport.py, EqM-GD
sampler t=0): SOLVES mazes 0.99 in-dist / 0.88-0.93 OOD (generalizes 4x larger) vs random
0.00 = POSITIVE CONTROL PASS. (3) metacognition: descent-shape probe (reuses
separability feature_groups) predicts maze-solve failure de-conf AUROC 0.67-0.76;
best-of-R=4 probe-restart > random @equal NFE on 2 seeds x 2 tiers, Δ+0.10..+0.22
valid-rate, 46-59% oracle. HARDER tier (c7 77% fail) -> higher AUROC (0.76) + bigger
rescue (+0.22) = difficulty-scaling story. Recovers MORE oracle than image best-of-R (14%)
b/c exact labels + dynamics-detectable failure. Files: gen_maze_data.py/eqm_maze.py/
maze_metacog.py + STEP2/STEP3_RESULTS.md + MAZE_EQM_PLAN.md (data gitignored, reproducible).
CPU-only, ~90s train. Caveat: new maze-EqM not IN-1K ckpt; claim="metacog works on real
trained EqM planner". OPEN: inpainting/translation rung (Yilun call).

**DECISION-GRADE @SCALE (2026-06-14, PUSHED main commit bc60fe1):** Both Yilun
questions ANSWERED. (1) CONSISTENT — 50k probe-gated best-of-R, 3 parallel seeds
(jobs 22931315/323/328 gpu 4xA100): vanilla->probe 28.20->26.21 / 27.78->25.95 /
27.83->26.04, mean **Δ1.87±0.11 FID, 95% CI ±0.12 excl 0, probe<vanilla ALL seeds**.
(2) Online equal-NFE sampler WORKS @15k (job 22975626): vanilla 29.55 (sanity OK vs
31.41) / random-restart 29.76 / probe-restart **28.51 (Δ1.24 equal-NFE)** / oracle
23.32, recovers 19% — TRUE online metacognition (restart probe-flagged mid-flight,
beats random-restart at identical NFE), mock->512->15k all consistent. Lineage stable:
best-of-R Δ 15k 1.69 -> 50k 1.87±0.11. Maze planning (CPU) probe 0.928 vs random 0.794
@equal compute (Yilun-suggested, general mechanism transfers). Claims #3/#4/#5 upgraded
to decision-grade in PAPER_CLAIM_STATUS; results in `results/SCALE_RESULTS.md`+FINDINGS§6.
PARTITION LESSON: seas_gpu est-start was ~12h out; moved 50k to `gpu` (full A100-sxm4,
NOT MIG -> 4-GPU DDP NCCL-safe; MIG risk is gpu_requeue-only) -> ran in ~2-3h. Per-seed
parallel jobs + afterok agg beats serial seed-loop. OPEN (Yilun call): inpainting/
translation rungs B/C; maze-depth. SSH session dropped 2x mid-run (2FA re-bootstrap);
jobs unaffected.

**PHASE 0-4 FOR YILUN (2026-06-14, branch `eqm-trajectory-metacognition`, commit
08a0682, NOT pushed):** Decision-grade end-to-end. (1) PARTIAL-probe `partial_probe.py`:
failure detectable as EARLY as step 100/249 (de-conf AUROC 0.814 ~= 0.818@end) ->
ONLINE-VIABLE. (2) `online_adaptive_sampler.py` true equal-NFE metacognition sampler
(restart flagged hi-risk mid-flight vs random subset, identical NFE) — MOCK PASS (probe
0.212 vs random 0.385 @ identical NFE 1294800); no-signal control confirms no NFE
artifact; real-scale `online_adaptive.sbatch`+`fid_online_agg.py` CLUSTER-GATED. (3) MAZE
PLANNING (Yilun-suggested) `maze_planning.py` FULL RUN 360 eval: energy-descent path
planner w/ real spurious minima, dynamics-probe branching = probe 0.928 vs random 0.794
valid-rate @EQUAL compute (oracle 0.944, 89% recovered, +0.228 hardest tier) — GENERAL
dynamics-metacognition mechanism TRANSFERS. Caveat: toy detection trivial (AUC~=1.0) vs
EqM 0.82; transferable claim = the ACTION not AUC. (4) 50k×3-seed consistency harness
`consistency.sbatch`+`consistency_agg.py` (probe_gated `--seed-offset`) — CLUSTER-GATED,
the #1 next run (turns 15k single-seed -> mean±CI). Docs: `RUN_INDEX.md`,
`YILUN_UPDATE.md` (asks: prioritize maze over inpainting?), `PAPER_CLAIM_STATUS.md` (claim
ledger: energy DEAD, dynamics SUPPORTED, FID-gain supported@15k-1seed, consistency NOT YET,
online mock-only, capabilities maze-only). Tests: test_posthoc + test_online ALL PASS.

**@SCALE CONFIRMED:** in-line restart sampler (probe-guided best-of-R=3) at 15k
trusted scale: vanilla 29.53 (≈baseline 31.41 -> sanity OK, pipeline valid), probe
27.84 (Δ1.69 better), oracle 17.75. Probe recovers 14% of oracle (< pool-rejection
45% — restart is a harder lever). Real controlled at-scale FID gain from a
trajectory-shape probe w/ NO image access. probe_gated_sample.py + fid_gated_agg.py
(+probe_gated.sbatch). Also built POSTHOC LADDER: robustness_analysis.py (label-sweep)
+ dynamics_probe.py (learned probe vs baselines) + sep_diag_posthoc.sbatch + test_posthoc.py
(3-case suite ALL PASS); real data endpoint WEAK / dynamics PROMISING 0.743; min-3-bin
norm-collapse guard. Local commit d71d4dc (NOT pushed; upstream + tangled 0a02e96).
OPEN (human): literal 50k multi-GPU; higher-R / threshold-adaptive rescue
(METACOGNITIVE_RESCUE_SPEC.md); PI framing. Full arc `FINDINGS.md`.

---

Separability diagnostic on vanilla EqM-B/2 80ep (FID 31.41 ckpt), 2026-06-13.
Tested the load-bearing assumption of a proposed "metacognition sampler" (detect
the low-grad-norm + high-energy "spurious minimum" cell of a norm×energy 2×2):
does any cheap scalar at the GD stopping point separate good vs garbage outputs
**independent of the gradient norm**?

3000 GD samples (η=0.003, cfg=1.0, 250 steps); independent labels = Inception
pool3 k-NN dist to 20k real ImageNet (crisp 25% tails, label-sanity s4=0.627);
5 candidate scores; **matched-norm control** = AUROC within grad-norm quantile
bins. Result (de-confounded, robust across fixed + τ∈{5,10,20}): dot energy
s1=0.609, path-integral s3=0.605 — both << the 0.80 action bar. **Decisive: the
dumb no-`f` latent-NN baseline s4=0.627 BEATS the energy scores** → EqM's energy
carries less good/garbage info than a trivial geometric probe. norm-coupled
s2/s5 ≈0.53 (chance). Verdict WEAK→**do not build the metacognition sampler**.
One retune used (CLAUDE.md cap): labels q0.40/0.30→0.25/0.25 moved KILL 0.582→
WEAK 0.609; signal real but faint, label-noise-limited at the margin.

Confirms the skeptic read: EqM energy is a **density** skeleton (good for OOD),
NOT a usable causal/quality axis. Cost ~1 GPU-hr, killed a weeks-long build.
Caveat: vanilla field only (explicit-energy EqM-E untested but is the paper's
fragile mode). Code: `projects/diff-EqM/experiments/separability_diagnostic/`
(4-stage: sample_with_logging→compute_quality_labels→compute_scores→analyze;
no-clone sbatch `sep_diag_local.sbatch`, rsync-deployed, NOT pushed — upstream
fix was in flight). Postmortem in that dir's POSTMORTEM.md.
