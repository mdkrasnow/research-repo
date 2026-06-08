# Experiment Queue — DG-ANM for EqM

## PUBLICATION GOAL
Targets: **NeurIPS 2026 workshop (Aug 29)** + **ICLR 2027 (~Oct 1)**. See `documentation/summer-2026-plan.md`.

Current phase: **1 — v10-only IN-1K seed 0 train RUNNING** (job 15638767 on seas_gpu, ~10h to completion).
Branch B-Both retired 2026-05-23 after CAFM-EqM Phase 1b FID 341.25 catastrophe (postmortem `postmortem-cafm-eqm-2026-05-23.md`).
v10 = PGD hard-example mining on EqM regression target. Single-objective, no discriminator, mining-based.


## CAPABILITY LADDERS (2026-06-07/08) — what does the v10 FID gain actually buy?

### Ladder v1 — KILLED (NULL). Postmortem `postmortem-capability-ladder-2026-06-07.md`.
Zero-shot clamped restoration (denoise/inpaint/transfer gray-lowres-blur-crop): v10 ~= vanilla
at noise level. Only falsifies NAIVE zero-shot conditional restoration. NOT "ANM learned nothing".

### Ladder v2 — COMPLETE (A-F). Doc `capability-ladder-v2-2026-06-07.md`. Memory diff_eqm_capability_ladder_v2.
Frozen ckpts, vanilla-s0 vs v10-s1 λ0.3. Verdict: v10 FID gain IS real+behavioral, BOUNDED to near-manifold quality.
- A gain-loc (Exp3): quality+class-adherence+coverage, recall flat, 91% classes ↑. POSITIVE.
- B hard-class: HARD ↑ 1.34x feat-dist / ~1.9x adherence vs easy. POSITIVE.
- D sampler sweep (job 19948994, 96 cells): no collapse; converged nfe>=100 −2.5..−3.2 FID; ~2.5x sample-eff
  (v10@nfe100 <= van@nfe250); NOT robust at starved nfe<=25. WEAK-POSITIVE.
- C rescue / E swap / F edit: NULL (no rescue, not splice-localizable, label-switch inert both arms).
Mechanism: PGD mining sharpens field NEAR data manifold in weak/hard-class regions. No far-from-manifold capability.
Paper: claim quality+hard-class+sample-efficiency; DO NOT claim editing/repair/low-NFE-robustness.

### Capability-ladder NEXT ACTIONS
1. **AWAIT PI decision** (needs_user_input set; pi-updates.md 2026-06-08 draft): accept framing for workshop draft,
   OR 3-seed-confirm A/B/D before locking claims. v10 λ0.3 seed0 ckpt was PRUNED → 3-seed would need regen/re-fetch.
2. If accepted → fold A/B/D into workshop draft "analysis" section; cite C/E/F nulls as honest bounds.
3. Revisit inpainting/outpainting/translation ONLY if a trained-conditional-head extension is authorized (out of current scope).
4. No long random-control train — gated, never triggered (checkpoint signal A/B/D is eval-only).
Infra kept: experiments/eval_trajectory.py (rescue/swap/edit), eval_capabilities.py (restore mode), exp1 sweep.


## Top-of-queue (Phase 1 → Phase 2)

1. **WAIT** for v10 train 15638767 to complete (step ~285K of 380K at last check; ETA ~10h on seas_gpu, 48h cap). Auto-pruner 15933157 keeps quota in check.
2. **ON COMPLETION** → `bash projects/diff-EqM/experiments/cafm_eqm/submit_v10_phase1_fid.sh` (50K-sample FID on latest ckpt). Gate: **FID ≤ 30.41** (vs vanilla 31.41).
3. **IF GATE PASS** → `bash projects/diff-EqM/experiments/cafm_eqm/submit_v10_phase2_seeds.sh` (seeds 1+2 80ep each). Phase 2 gate: 3-seed Welch t p<0.05, mean ≥ 1 FID gain.
4. **IF GATE FAIL** → 1 retune of λ ∈ {0.03, 0.3, 1.0} per CLAUDE.md, then kill direction → propose v11 (Briglia equivariant fallback, sketch in `documentation/v11_fallback_sketch.md`).
5. **PI update trigger** on Phase 1 gate result (drafted in `pi-updates.md`, user-send only).
6. **Phase 3** (gated on Phase 2 PASS): scaling curves S/2, B/2, L/2 on IN-1K.
7. **Phase 4** (gated on Phase 3): SiT transfer ≥ 0.5 FID.
8. **Phase 5** (gated on Phase 4): workshop draft ready by 2026-08-22 (7-day buffer to deadline).

## Analysis experiments (mechanism / robustness)
- **Exp 1 — NFE/sampler robustness** (`experiments/exp1_sampler_robustness/`; findings `documentation/exp1-sampler-robustness-findings.md`) — RUNNING: Smoke A+B PASS. Full 5k run 17828606 died 58/80 (SIGPIPE, debugging.md); 57 cells recovered. **Interim (gd only): ANM wins ~2 FID ONLY in converged regime (nfe≥100), loses slightly at low-NFE → improves final quality, NOT low-NFE robustness (partial/negative for robustness hypothesis).** Merge/resume 18003763 RUNNING to finish ~23 missing cells (all anm_ngd). NEXT: full-80 AUC verdict → if holds, 50k paper-grade resume. Fixes: free-port (8aa5308), SIGPIPE-safe logging + resume-trusts-FID (c88ffbe).
- **Exp 2 — off-trajectory field robustness — ✅ DONE 2026-06-01** (`experiments/diagnostics/offtraj_field_robustness.py`; results `documentation/exp2-offtraj-field-robustness-results.md`; data `results/diagnostics/offtraj_{random,sampler}/`; jobs 17788287+17788329 both exit 0). RESULT: ANM field-robustness mechanism CONFIRMED at IN-1K B/2 (5120 latents, paired bootstrap CI). ANM lower MSE/higher cosine at every radius (all SIG); gap widens off-traj (random dMSE −0.0164→−0.0277 r0→r0.1); **largest at real v10-mined δ (dMSE −0.0368 [CI −0.0396,−0.0340], dCos +0.00109)**. Norm-reshaping RULED OUT (norm_ratio 0.730 both); peaks mid-t~0.55–0.75, not t→1. Effect SMALL (0.2–0.4% rel) — consistent with B/2 NULL capability eval; may amplify at XL/2. PI update drafted (do-not-send) in `pi-updates.md`. Logged to `results_variants.tsv`.
- **Exp 3 — fidelity-diversity & mode coverage — ✅ DONE 2026-06-05** (`experiments/exp3_fidelity_diversity/`; results `documentation/exp3-fidelity-diversity-results.md`; data `results/exp3_metrics_out/` + gen on holylabs `mkrasnow_eqm/exp3/`; jobs 18964347 ref + 18964349 anm-gen + 19120911 metrics, exit 0). VERDICT **STRONG_SUCCESS — no diversity tax**. vanilla vs ANM l03, EqM-B/2 80ep IN-1K-256, 49996 identical ids, gd/250/cfg1.0/EMA, fixed seeded ref. FID 26.88 vs 31.27 (−4.38, disjoint 95% CIs). **recall FLAT 0.7185→0.7193 (diversity preserved)**, coverage +0.072 (0.443→0.515), density +0.044, precision +0.023, KID −0.0057. **Weak-class bottom-quartile FID −5.61 (62.80→57.19, weak gain MORE)**; classifier TV 0.181→0.162; cond-top1 +0.050; 91% classes improve. Closes "you just sharpened samples" reviewer attack. Caveat: single seed — Phase 2 3-seed Welch t still required for paper-final. Logged to `results_variants.tsv`; PI update drafted (do-not-send) in `pi-updates.md`.

## Capability probes (elevate above workshop — "new feature" claims, frozen-ckpt, no retrain)
Goal: find a categorical capability ANM gives EqM that vanilla lacks (not incremental FID). Each
pre-registered with mechanism arg + kill rule. Arms: vanilla 31.41 / v10 λ=0.1 29.01 / v10 λ=0.3 27.09.
- **C1 — Inference-compute scaling** (`documentation/c1-inference-compute-scaling-proposal.md`) —
  PROPOSED. Does ANM keep improving with more solver steps where vanilla plateaus/overshoots? Reframes
  Exp 1 data (energy `E` + `‖∇E‖` traces via get_energy). **Minimal test = re-analyze existing Exp 1
  80-cell CSVs, ZERO new compute.** Promote if vanilla FID-vs-NFE turns up while ANM monotone, dose-ordered.
- **C2 — Restoration / corrupted+OOD-init robustness** (`documentation/c2-restoration-init-robustness-proposal.md`) —
  PROPOSED. Descend EqM from corrupted init; does ANM advantage GROW with corruption severity (the axis
  the NULL B/2 inpaint eval collapsed)? Minimal test ~0.5 GPU-day (Gaussian-noise init, 3 σ, N=512).
- **C3 — OOD detection via energy** (`documentation/c3-ood-energy-detection-proposal.md`) —
  PROPOSED. Use scalar `E` / `‖∇E‖` as OOD score → AUROC. Discriminative metric orthogonal to FID.
  Minimal test <0.5 GPU-day, no sampler/decode. CHEAPEST + most legible win if it holds.
- Skeptic flag (all three): B/2 capability eval was NULL + Exp 2 effect small (0.2-0.4% rel). Capability
  separation likely needs λ↑ and/or XL/2 scale. Minimal tests are filters; do C1 re-analysis + C3 first.

## In-flight
- 15638767 v10 IN-1K seed-0 train (seas_gpu, RUNNING)
- 15933157 ckpt auto-pruner (shared, RUNNING)

Historical top-of-queue items preserved below for reference but no longer active.

## Research Question
Does differential-geometry-guided adversarial negative mining improve EqM's equilibrium landscape, reduce spurious equilibria, and improve optimization-based sampling?

## Hypotheses to Test
1. Baseline EqM-S/2 has measurable short_horizon_recovery_distance on CIFAR-10
2. Normal-space perturbations produce harder negatives than random perturbations
3. Adversarial search (PGA on mining objective) finds harder negatives than random normal-space
4. Trajectory failure term (L_traj) adds signal beyond field norm (L_weak)
5. Combined DG-ANM improves recovery distance vs baseline

---

## READY

### Q-001: Baseline EqM-S/2
- Hypothesis: Establish baseline performance — no mining
- Config: `configs/baseline.json`
- Resources: 1x A100, ~5 min (1 epoch CIFAR-10)
- Priority: HIGH (must run first, establishes baseline metric)
- Notes: This is the autoresearch baseline iteration

### Q-002: DG-ANM Basic (normal + weak)
- Hypothesis: Normal-space perturbations with L_weak improve recovery
- Config: `configs/dganm_basic.json`
- Resources: 1x A100, ~8 min (mining overhead)
- Priority: HIGH (first DG-ANM test)
- Notes: Simplest mining — validates geometry matters before adding complexity

## IN_PROGRESS
(none)

## DONE
(none)

## FAILED
(none)
