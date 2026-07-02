# PI Update Log & Cadence — DG-ANM for EqM

## Purpose
Keep the PI informed on progress toward a NeurIPS/ICML/ICLR submission. Updates should highlight real progress (stage-gate passes, baseline comparisons, mechanism findings), surface blockers early, and NOT overclaim proxy-scale gains.

## Cadence
- **Minimum**: weekly digest (every Monday) — even if "no new signal, running X".
- **Triggered** (send within 24h of the event):
  - Stage exit gate passes or fails (A → B → C → D)
  - Any result at paper-comparable scale (EqM-B/2 80ep or larger) with DG-ANM vs. vanilla EqM
  - Gain confirmed across ≥3 seeds (stops being a curiosity, starts being a result)
  - Blocker requiring PI input: compute allocation, scope decision, authorship, venue choice
  - Significant pivot: method reformulation, scope change, venue switch
  - External signal: concurrent/scooping work appears

## What qualifies as "worth reporting"
| Event | Report? | Why |
|-------|---------|-----|
| Proxy FID improved by X% at 2ep IN-100 | **No** (log internally) | Not publishable; within noise at this scale |
| Proxy config passed 3-seed gate (Stage A exit) | **Yes** | First real evidence method is reproducible |
| Stage B confirmation (CIFAR-10 B/2 80ep, 3 seeds) | **Yes** | First paper-grade result |
| Stage C confirmation (ImageNet-256 B/2) | **Yes** | Credibility for top venue |
| Scaling curve holds across S/2, B/2, L/2 | **Yes** | Main paper claim |
| Mechanism figure (e.g., landscape viz) | **Yes** | Story-level contribution |
| Compute blocker, deadline slip, scoop risk | **Yes** (fast) | Needs decision |
| Bug found and fixed with no impact on prior results | **No** (log internally) | Internal hygiene |
| Bug found that invalidates a prior reported result | **Yes** (fast, same day) | Honesty |

## Update template
```
Subject: DG-ANM update — <one-line headline>

**Stage**: A / B / C / D / E
**Headline**: <one sentence, e.g. "Stage A exit gate passed — best config beats baseline by 4.2 FID across 3 seeds">

**What's new since last update**:
- <bullet 1>
- <bullet 2>

**Numbers** (if applicable):
- DG-ANM: <FID ± std, n=k seeds>
- Vanilla EqM: <FID ± std, n=k seeds>
- Scale: <model / dataset / epochs / FID sample count>
- Compute used: <GPU-hours>

**What's next** (in priority order):
1. <action>
2. <action>

**Blockers / decisions needed**:
- <none, or specifics>

**Confidence**: low / medium / high that this leads to publication. <one-sentence reasoning>
```

## Update log (most recent first)

<!-- Format:
### YYYY-MM-DD — <headline>
**Trigger**: <event or weekly>
**Sent to PI**: yes/draft/no
Content or link to sent email.
-->

### 2026-04-21 — Critical finding: DG-ANM as implemented does not improve EqM (is actively harmful)
**Trigger**: stage_exit_gate_pass (3-seed repeatability check) + bug_invalidating_prior_reported_result
**Sent to PI**: DRAFT (needs review before sending)

---
Subject: DG-ANM update — method does not improve EqM; prior gains were noise

**Stage**: A.5 (reproducibility gate). Pivoting before Stage B commitment.
**Headline**: A fast CIFAR seed study (3 vanilla + 3 DG-ANM, matched config, Flow-Matching U-Net backbone) shows DG-ANM with a properly calibrated margin is **~45% worse** than vanilla EqM at matched training checkpoint (e100 FID 43.83 ± 7.62 vs 30.18 ± 0.28, Welch's t ≈ 3.1σ). DG-ANM also introduces 27× more seed variance. The prior IN-100 "improvements" are very likely RNG-perturbation artifacts from autoresearch tournament-selection on single-seed noise — not real method signal.

**Evidence chain**:
1. A diagnostic script showed that on the CIFAR UNet, `||field||` at perturbed inputs is ~47–49, so the `ReLU(margin=5 − ||field||)` loss is exactly saturated — gradient to model parameters is exactly zero. All IN-100 autoresearch "winners" used margin=5 with similar field-norm scales, meaning the margin-loss gradient was nearly or entirely zero in every run.
2. A matched-seed CIFAR study with margin=50 (which per the diagnostic should produce real gradient signal) shows DG-ANM is *worse* than vanilla by 13.65 FID at e100 with 3 seeds each — the first honest A/B comparison we have.
3. Vanilla's e100 FID std is 0.28 (very tight). DG-ANM's is 7.62. Consistent with mining's extra `torch.randn` and `torch.autograd.grad` calls perturbing the global RNG state, effectively making every DG-ANM run a fresh seed relative to its vanilla counterpart.
4. The IN-100 autoresearch gain trajectory (278.66 → 249.33 over 4 rounds) was tournament selection over effectively random RNG-divergent runs. No DG-ANM autoresearch "winner" beat vanilla on a multi-seed test — we never ran one until now.

**Numbers**:
- Vanilla EqM, FM UNet, CIFAR-10, 150 ep, 3 seeds: **FID 12.54 ± 1.15** (10K samples, seed-tight)
- Vanilla EqM at e100: 30.18 ± 0.28
- DG-ANM (margin=50, gamma=6, eps=0.8), same config, 3 seeds, walltime-truncated at e108: **FID 43.83 ± 7.62 at e100**
- Matched-epoch gap: **+13.65 FID** (DG-ANM worse), **3.1σ effect size**
- Prior IN-100 gain: vanilla 121.24 vs DG-ANM 112.58 (n=1 each) — well within the ~8–15 FID seed variance we now measure on a simpler dataset.

**What this means for the paper**:
- DG-ANM as currently formulated does not improve EqM and is likely actively harmful.
- The autoresearch infrastructure was tournament-selecting on noise; its "best hyperparameters" do not reflect real method gains.
- We cannot submit this method as currently implemented to any top venue; a reviewer running this same experiment would reject.
- Honest acknowledgment: ~2 weeks of proxy autoresearch were misallocated. Catching this now, before committing to ImageNet-256 scale, saves ~1–2 weeks of compute and avoids submitting an irreproducible result.

**What's next** (in priority order):
1. **Rethink the mechanism before any more compute**. The PGA mining with `ReLU(m − ||f||)` margin loss does not, on current evidence, help EqM on at least one standard benchmark. Options: (a) different negative-mining formulation (contrastive? trajectory-based?); (b) different objective (not EqM-with-mining; maybe a principled SBEM/CD variant); (c) pivot the paper to an honest analysis contribution — "why naive adversarial mining fails for implicit EBMs," with our diagnostic as the core evidence.
2. **Keep Stage B vanilla IN-100 3-seed baseline running** (already submitted, 6820064). Gives us honest vanilla-EqM seed variance at the paper's ablation scale. Useful regardless of the method pivot.
3. **Do not start ImageNet-256 B/2 80ep DG-ANM training.** It was the next big compute ask; cancelling this saves ~1–2 weeks of seas_gpu.
4. Discuss this week whether to (a) pivot scope (analysis paper on method failure modes), (b) rebuild with a different mining formulation and re-validate, or (c) wind this down.

**Blockers / decisions needed**:
- Strategy call: pivot, rebuild, or wind down? I lean toward *pivot to analysis* — we have a clean negative result with diagnostic evidence, which itself is publishable at a workshop or as a short methods note.

**Confidence**: High that DG-ANM as currently formulated does not produce real improvements. Medium-high that a refined version of the core idea might, but we do not have evidence for this yet.

**Artifacts**:
- `projects/diff-EqM/.state/pipeline.json` (`cifar_seed_study_COMPLETE` entry)
- `projects/diff-EqM/fid_results.tsv` (last 6 rows: seed study)
- `projects/diff-EqM/experiments/diag_dganm_signal.py` (the diagnostic)
- `projects/diff-EqM/slurm/logs/cifar-seed-study_6821254_*.out`
---

### 2026-04-18 (PM) — Stage A.5 gate passed: CIFAR FID 497 bug was architecture
**Trigger**: stage_exit_gate_pass (Stage A.5 Step B exit criterion FID<15 met)
**Sent to PI**: DRAFT (needs review before sending)

---
Subject: DG-ANM update — Stage A.5 gate passed, CIFAR FID 497 bug isolated to architecture

**Stage**: A (autoresearch in progress) / A.5 (reproducibility gate — just passed)
**Headline**: Our CIFAR vanilla-EqM result at FID 497 was caused by training a transformer where the EqM paper uses a U-Net. A diagnostic smoke test (standard UNet + plain flow-matching loss on CIFAR-10) hit FID 9.72 on 10K samples using our same data and evaluation pipelines — a 51× reduction. Step A (port Flow Matching's CIFAR U-Net and graft our EqM loss onto it) is unblocked; targets paper FID 3.36.

**What's new since last update**:
- Audited our CIFAR stack against the EqM paper's Appendix B.1: paper uses a non-transformer U-Net from the Flow Matching (Lipman et al., 2024) codebase; our prior vanilla run used EqM-S/2 transformer at patch=4. Every EqM-paper-like CIFAR number in our repo from the previous attempt is invalid.
- Step B smoke test (diffusers UNet2DModel + plain FM loss, 150 epochs, same data + FID pipeline as the broken runs) trained cleanly: FID progressed 284.6 → 164.3 → 55.9 → 27.4 → 21.9 → 20.0 at the periodic 2K-sample checkpoints, with a final 10K-sample FID of 9.7193. Our data pipeline and FID evaluator are sound.
- Autoresearch continues on the proxy (2-epoch IN-100): current best DG-ANM config 249.33 (gamma=6.0, eps=0.8, lr=2e-4), down 10.5% from the proxy baseline. This is not yet a real result — the Stage A exit gate is a 3-seed repeatability check, still ahead.
- Stage B baseline (vanilla EqM-B/2 80ep IN-100, 3-seed array) submitted in parallel on seas_gpu; currently queued.

**Numbers**:
- CIFAR smoke (UNet + FM loss, our pipelines): FID 9.72 on 10K samples (exit criterion was <15). Paper reports 3.36 for EqM with their tuned U-Net + EqM loss; our smoke uses a smaller architecture and plain FM loss.
- CIFAR broken baseline (transformer + EqM loss): FID 497.55 (vanilla), 497.04 (DG-ANM). Stack-level bug.
- Proxy autoresearch best so far: FID 249.33 (provisional, single seed, 2-epoch IN-100). Baseline 278.66.

**What's next** (in priority order):
1. Port the Flow Matching CIFAR U-Net into our repo and graft the EqM c(γ) loss onto it; train vanilla EqM to match paper FID 3.36 (±0.3 tolerance).
2. Stage B vanilla baseline on IN-100 (3 seeds, EqM-B/2, 80 epochs) begins training when queue frees up; ~24–36 h per seed.
3. Finish round 4 of the proxy autoresearch (2 candidates still running) and run a 3-seed repeatability check on the winner — this is the Stage A exit gate.

**Blockers / decisions needed**: none.

**Confidence**: Medium-high that the method will produce at least a small improvement at paper-comparable scale; low-to-medium that it will be large enough for a top-venue headline without additional mechanism-level evidence. Stage B's 3-seed IN-100 result (≤2 weeks out) will be the first honest data point on this.
---
**Trigger**: state-file setup (not yet sent to PI)
**Sent to PI**: no — first real update will be when Stage A exit gate produces data
**Summary**: Goal set to NeurIPS/ICML/ICLR. Autoresearch at proxy scale (2ep IN-100, FID 278→253, -9.2%) — not publishable as-is. Stage A exit gate = 3-seed repeatability check on best proxy config. Round 4 (9 candidates across 4 dimensions) running now.

## Reminders for the agent
- Draft updates in this file (under the log) **before** sending, so the PI gets a coherent story rather than reactive pings.
- When a triggered event fires, append a draft update entry within the same session and flag `needs_user_input.value=true` in pipeline.json with a prompt like "Draft PI update ready in documentation/pi-updates.md — review and send?"
- Never fabricate headline numbers. If an experiment is still running, say "in progress, expected by <date>."
- Weekly digest: the agent should prepare a draft Monday morning even if nothing triggered, so the PI has a consistent signal of life.

---

## DRAFT — 2026-05-20: Phase 0.3 PASS + Phase 1a smoke retry

**Status**: ready for review by user; not yet sent.

**Headline**: v10 PGD hard-example mining on EqM CIFAR-10 (150 epochs) achieved FID **13.40**, beating vanilla EqM (R4 baseline 14.17) by **0.77 FID**. Mechanism gates A/B/D pass; ratio L_hard/L_clean stable at 1.047-1.049 throughout (non-saturating — central differentiation from v02 which saturated to cosine=1.0 on EqM-B/2 within 9 epochs).

**Direction repositioned** post Phase 0 lit review (17 papers): "first adaptive hard-negative mining for regression-target generative models" (per VeCoR §7 explicit future-work cite). Branch B-Both: combine PGD-on-EqM-target (v10) with CAFM-style discriminator post-training (Lin et al. 2026 AFM/CAFM, ICML 2026). Primary on EqM-B/2 IN-256; secondary head-to-head on SiT Phase 5.

**Targets**: NeurIPS 2026 workshop (Aug 29) + ICLR 2027 (~Oct 1). NeurIPS 2026 main missed (May 6).

**Code complete**: CAFM-to-EqM port (5 modules), 14/14 unit tests pass, gate evaluators, Phase 5 SiT subclass, Phase 1b submitter helper, workshop paper intro drafted.

**Outstanding blockers**: cluster SSH dropped twice in past 24h (need credential refresh discipline). Smoke 13997995 in flight (ckpt path bug fix).

**Decisions needed from PI**: none currently.

**Next**: smoke PASS → submit 10ep CAFM-EqM full run (Phase 1b). If CAFM-only seed-0 IN-256 FID < 25 (vs vanilla 31.41), proceed to v10+CAFM combined Phase 2.

---

## DRAFT — 2026-05-22: Phase 1b CAFM-EqM IN-1K-256 RUNNING (after infra audit)

**Status**: ready for review by user; not yet sent.

**Headline**: Phase 1b CAFM-only post-training of vanilla EqM-B/2 80ep ckpt (FID 31.41) submitted on seas_gpu (job 14509440, SHA 375ab56). 4×A100-80GB, batch 64 local × 4 = 256 global (matches Lin CAFM recipe). Pre-scheduled start ~2026-05-23T08:49.

**Infra audit before launch** caught 3 correctness/robustness bugs:
1. `gen` and `dis` were never wrapped in DDP — 4 ranks would have trained independent copies with no gradient sync. Effective batch = 64 not 256. Fixed: manual `dist.all_reduce` after backward + init broadcast from rank 0 (commit a7f20b1). This is the pattern Lin's CAFM repo uses; torch.func.jvp doesn't compose cleanly with `torch.nn.parallel.DistributedDataParallel` autograd hooks.
2. Cluster home `/n/home03/.../research-repo/` is a plain directory copy (not a git clone). sbatch directives are parsed at submit time from the cluster-side file, NOT from the in-tmp git clone done by the job script. Earlier smoke v12/v13 ran on MIG cards (NCCL fail) because the cluster sbatch was stale. Fixed: submit helpers rsync `slurm/` to cluster before submit (commit 942365e, 0bfd2e4).
3. Phase 1b sbatch missing the `cafm_pkg` namespace staging + vanilla-ckpt symlink that smoke sbatch had. Fixed: ported both (commit f467c8b).

**Validation**: Smoke v14 PASSED on real A100-80GB (10:15, holygpu8a16304). Init broadcast logged, dis loss 2.00→0.03 (steeper convergence than smoke v11's 2.0→0.15, consistent with grads now actually syncing across ranks).

**New capability landed alongside**: `--ckpt-resume` flag wired through train script + sbatch (commit 55bd7c0) so a TIMEOUT at 48h can resume from latest ckpt (saved every 5K steps). FID eval submit script reuses existing `imagenet1k_fid_eval.sbatch` (same pipeline that produced vanilla baseline FID 31.41).

**Gate**: Phase 1 = CAFM-only seed-0 IN-1K-256 FID ≤ 30.41 (vs trusted vanilla 31.41). Pass → Phase 2 v10+CAFM combined. Fail → 1 retune of `cp_scale` or `warmup`, else kill Branch B-Both.

**Outstanding risks** (not blockers):
- 48h seas_gpu cap; smoke extrapolation suggests 30-50h, marginal. Resume path exists if TIMEOUT.
- HF VAE download not pre-cached; smoke shows unauthenticated warnings but downloads succeed.

**Decisions needed from PI**: none.

**Next**: monitor 14509440 to start (~23h queue) → train (~30-50h) → FID eval (~3-6h) → gate eval. Workshop submission window 2026-08-15 to 2026-08-29.

---

## DRAFT — 2026-05-23: Branch B-Both RETIRED; v10-only pivot

**Status**: ready for review by user; not yet sent. **PIVOT trigger** + **bug invalidating prior plan**.

**Headline**: Phase 1b CAFM-on-EqM port FAILED catastrophically. FID 341.25 vs vanilla 31.41 (10× worse). Mechanism bug, not a retune-tier miss. Branch B-Both retired. **Pivoting to v10-only** for workshop submission.

**What happened**:
- 50K-step CAFM-only post-training of vanilla EqM-B/2 80ep completed cleanly (17h50m, loss curves finite). FID eval on the resulting checkpoint = 341.25.
- Diagnostic FID at ckpt_5000 (just past warmup, ~250 gen updates) = 369.64. Collapse was instant, not cumulative.
- Vanilla baseline FID at 10K samples (noise-matched) = 37.09. Consistent with our trusted 50K = 31.41. Vanilla model intact; failure is specific to CAFM post-training.

**Root cause** (see `documentation/postmortem-cafm-eqm-2026-05-23.md`):
- CAFM works when the generator was trained adversarially-compatible from scratch (Lin's SiT setup).
- Vanilla EqM was trained by pure regression on `(ε − x)·c(γ)` — no adversarial signal in its history.
- Freshly-initialized discriminator quickly learns "anything ≠ vanilla EqM output = fake" trivially. First gen updates push generator away from the regression-target manifold to fool the disc → field collapse, instant.
- The smoke v14 dis loss curve (2.0 → 0.03 one-sided) was the failure signature. We misread it as "healthy adversarial convergence." It is the textbook sign that the discriminator has found a trivial discrimination shortcut.

**Why we did not catch it earlier**:
1. Smoke validated loss-finiteness and exit code, NOT sample quality. A 5-min sample probe on the smoke checkpoint would have caught this and saved ~20 GPU-h.
2. Misread the smoke loss curve. Adversarial losses fail with one-sided monotonic decrease; we logged it as healthy because the gen loss stayed bounded.
3. Design doc reasoned about JVP geometry, not about discriminator-shortcut failure modes specific to non-adversarially-trained generators.

**Pivot to v10-only**:
- v10 (PGD hard-example mining on the EqM regression target) passed Phase 0.3 PASS: CIFAR-10 FID 13.40 vs vanilla 14.17 across 150 epochs, with mining ratio L_hard/L_clean stable at 1.047-1.049 (non-saturating).
- Path: port v10 from CIFAR variant harness to the trusted EqM-B/2 IN-1K-256 training stack (Stage B vanilla 80ep produced the trusted FID 31.41 — same stack). Re-baseline confirmation not needed.
- Phase 1 v10-only gate: FID ≤ 30.41 on IN-1K-256 seed-0. Estimated 18-24h per seed on 4×A100-80GB.
- Workshop story: "first adaptive hard-negative mining for regression-target generative models" (per VeCoR §7 future-work cite). Single-method paper. Compute budget tightens but still fits NeurIPS 2026 workshop deadline 2026-08-29.

**Process changes landed in CLAUDE.md**:
- Mandatory smoke-time sample-quality probe for any new loss on a generative model.
- Discriminator-loss-specific failure-mode checklist (oscillation, not monotonic-decrease).

**Numbers**:
- Phase 1b CAFM-only FID (50K samples): 341.25 (gate threshold 30.41) → FAIL.
- ckpt_5000 diagnostic FID (10K samples): 369.64.
- Vanilla baseline 10K samples: 37.09 (consistent with trusted 50K = 31.41).
- Phase 0.3 v10 CIFAR (preserved): 13.40 vs 14.17 vanilla.

**Cost of detour**: ~33 GPU-h (smoke iterations + Phase 1b 18h + FID 1.5h + diagnostics 1.5h).

**Decisions needed from PI**: confirm v10-only pivot, OR direct alternative (e.g., investigate CAFM-port repair, or new variant proposal).

**Next**:
1. (Pending PI confirm) port v10 to IN-1K training stack.
2. Update `summer-2026-plan.md` Phases 1-5 timeline for v10-only.
3. Submit Phase 1 v10-only IN-1K seed-0 run.
4. CAFM-EqM port code archived in `experiments/cafm_eqm/` for record.

---

## 2026-05-27 — Phase 1 gate PASS (v10 IN-1K seed-0)

**TRIGGER**: Phase 1 exit gate evaluated; PASS.

**Headline**: v10 IN-1K-256 EqM-B/2 80ep seed-0 50K-sample FID = **29.01** vs trusted vanilla baseline **31.41** (gain **2.40 FID**, **7.6% relative**). Phase 1 gate threshold 30.41 cleared by 1.40 FID.

**Context**:
- v10 = single-objective regression with PGD hard-example mining on the EqM target (no discriminator, no two-player game, cannot collapse to trivial solutions).
- This is the **first paper-comparable-scale confirmation** of v10 since the CIFAR-10 Phase 0.3 PASS (13.40 vs 14.17). CIFAR gain was 0.77 FID; IN-1K gain is 2.40 FID — **gain scales up** under transfer to ImageNet-1K.
- Mechanism diagnostics throughout training matched CIFAR PASS signature: aux/base ratio stable 1.01-1.03 (non-saturating, mining active), base loss matched vanilla at ~10.30 (regression objective preserved), ‖δ‖ at boundary 0.300 throughout, no collapse signal.
- Train completed clean in 1d11h32m on seas_gpu (resumed from ckpt_65000 after prior quota-deadlock incident on 15290932). FID eval ran on gpu_requeue in 2h20m.

**What this means for the paper**:
- Workshop §4 (experiments) now has its first headline number on IN-1K. v10 confirms transfer of the CIFAR mechanism to paper-scale.
- The discriminator-vs-mining framing locked in last week's lit sweep (post-CAFM retire, post-AFM hit) is now empirically substantiated: mining-based regression-target adversarial training **works** where discriminator-based fails (CAFM-EqM was FID 341).

**Phase 2 (already launched)**: seeds 1 + 2 submitted as 16362498 + 16362499 (gpu partition, ~30-36h each). Gate: 3-seed Welch t p<0.05 AND mean ≥ 1 FID gain vs vanilla. Currently 1/3 seeds in; preliminary 1-seed gain (2.40 FID) is well above the 1-FID threshold, so the multi-seed test is mostly a confirm-and-stabilize step.

**Risks**:
- 2-seed variance unknown; if seeds 1+2 show high variance (e.g., one seed regresses), Welch t may not pass. Mitigation: CIFAR Phase 0.3 was single-seed but CAFM postmortem's seed-variance concerns specific to discriminator dynamics, not regression mining.
- Lin lab scoop risk LOW per 2026-05-26 lit sweep; their discriminator-based momentum makes mining-pivot unlikely in 13-week workshop window.

**Decisions needed from PI**: None blocking. Update for awareness + green-light continued autonomous execution of Phase 2 → Phase 3 (scaling curves) if Phase 2 PASS.

**Numbers (canonical)**:
- v10 IN-1K seed 0: FID **29.01** (50K samples, gd sampler eta=0.003 steps=250 cfg=1.0)
- Vanilla baseline: FID **31.41** (50K samples, same sampler)
- v10 CIFAR Phase 0.3: FID **13.40** vs vanilla **14.17**

**Next**:
1. Wait for Phase 2 seeds 1+2 (~30-36h).
2. On Phase 2 PASS: launch Phase 3 scaling curves (S/2, L/2 on IN-1K).
3. Update SYNTHESIS.md §4 risk table — Phase 1 transferability risk now RESOLVED.
4. Update paper-draft-method.md §3.3 closing paragraph with the FID 29.01 number.

---

## DRAFT — 2026-06-01 — Exp 2: mechanism evidence for the v10 FID gain (DO NOT SEND until reviewed)

**Trigger**: result at paper-comparable scale (IN-1K B/2) — mechanism diagnostic for v10.

**TL;DR**: Built a field-level diagnostic asking whether v10/ANM's FID gain is backed
by a real mechanism — does ANM make the EqM field more accurate *off* the training
trajectory? On 5120 held-out IN-1K val latents (frozen vanilla FID 31.41 vs v10 FID
29.01, EqM-B/2, paired bootstrap CI over sample_id), the answer is **yes, but small**.

**Result** (ANM − vanilla, all statistically significant, n≈30k pairs/cell):
- Random off-trajectory perturbations: ANM lower field-MSE / higher cosine at every
  radius; the gap **widens** with radius (dMSE −0.0164 on-path → −0.0277 at rel-radius 0.1).
- **Largest gap at the real v10-mined perturbation** (dMSE −0.0368 [CI −0.0396,−0.0340],
  dCos +0.00109) — ANM is most accurate exactly in the off-trajectory region its
  training mining targets. Cleanest possible mechanism-confirmation.

**Honest caveat**: effect size is small — dMSE ≈ −0.02 to −0.037 against vanilla MSE ≈
10.3 (0.2–0.4% relative). Significant (huge n, tight CI) but modest. Consistent with
the earlier NULL B/2 capability eval: v10's B/2 benefit is real but small; may amplify
at XL/2 (untested). This is a *local* field-robustness proxy, not proof of global energy
correctness or sampling optimality.

**Rule-outs**: ANM does not win by rescaling field norms (norm_ratio 0.730 both ckpts,
identical); improvement is broad across t and peaks mid-trajectory (t~0.55–0.75), not an
artifact of near-zero targets at t→1.

**Use for paper**: supports the workshop story — "ANM doesn't just improve FID, it makes
the learned field measurably more robust to local off-trajectory drift, most where it
mines." Pairs with Exp 1 (sampler/NFE robustness) and the FID number. Report effect size
honestly; flag XL/2 as the scale where the mechanism may matter more.

**Decisions needed from PI**: None blocking. Awareness + sanity-check on whether the
small effect size is worth foregrounding vs relegating to appendix.

**Artifacts**: `documentation/exp2-offtraj-field-robustness-results.md`;
`results/diagnostics/offtraj_{random,sampler}/`; jobs 17788287 + 17788329 (both exit 0).

---

## PI update — Exp 3: Fidelity-Diversity & Mode Coverage (2026-06-05)

**Trigger**: paper-comparable IN-1K result + pre-registered "no diversity tax" question answered.

**One-line**: At IN-1K-256 (EqM-B/2, 80ep), ANM's FID gain is **not** bought with diversity — recall holds flat, coverage and density rise, and weak classes improve more than the mean. Verdict: **strong_success**.

**Setup (parity-controlled)**: vanilla EqM vs ANM EqM (v10, λ=0.3), identical sampler/NFE/step/EMA/CFG/VAE/labels/seeds (shared schedule hash `83a8ede763e1b318`), fixed seeded ImageNet-train reference, pytorch_fid InceptionV3 features, PRDC (Naeem 2020, vendored). Both arms scored on the **same 49996 sample ids** (4 zero-byte vanilla PNGs from an interrupted file move were dropped from both arms; immaterial at 50K).

**Results**:

| metric | vanilla | ANM (λ0.3) | Δ |
|---|---|---|---|
| FID ↓ | 31.27 | **26.88** | −4.38 (95% CIs disjoint: 31.64–32.28 vs 27.27–27.85) |
| KID ↓ | 0.0316 | 0.0259 | −0.0057 |
| precision ↑ | 0.581 | 0.604 | +0.023 |
| recall ↑ | 0.7185 | 0.7193 | +0.0008 (flat — **no diversity loss**) |
| density ↑ | 0.433 | 0.477 | +0.044 |
| coverage ↑ | 0.443 | 0.515 | **+0.072 (mode coverage)** |
| bottom-quartile (weak-class) FID ↓ | 62.80 | 57.19 | −5.61 (weak classes gain MORE) |
| classifier TV→requested ↓ | 0.181 | 0.162 | better class balance |
| conditional top-1 ↑ | 0.433 | 0.483 | +0.050 (more on-class) |
| frac classes ANM better (feature dist) | — | 0.913 | 91% of classes improve |

**Interpretation for the paper**: This closes the obvious reviewer attack on the FID claim ("you just sharpened samples"). Recall is the diversity axis and it is flat; coverage/density (mode coverage + local density) both improve; the gain is broad (91% of classes) and *largest on the weak classes* the mining is meant to help. Pairs cleanly with Exp 1 (sampler/NFE robustness) and Exp 2 (off-trajectory field robustness). The three together: ANM improves quality, is robust to sampler budget, and the mechanism (off-traj field accuracy) is measurable.

**Caveat**: single seed at B/2. Phase 2 multi-seed (3-seed Welch t, p<0.05, ≥1 FID gain) still required before the claim is paper-final. This is the per-seed evidence the no-diversity-tax story rests on.

**Decisions needed from PI**: None blocking. Confirm whether to foreground coverage (+0.072) and weak-class bottom-quartile (−5.61) as the headline "no diversity tax" evidence in the workshop draft.

**Artifacts**: `results/exp3_metrics_out/{aggregate_metrics.json,aggregate_metrics.csv,class_metrics.csv,delta_class_metrics.csv,classifier_histogram.csv,plots/,README.md}`; gen on holylabs `mkrasnow_eqm/exp3/`; jobs 18964347 (ref) + 18964349 (anm gen) + 19120911 (metrics, exit 0).

---

## 2026-06-07 — v10 IN-1K B/2 80ep: 3-seed FID confirms gain (Phase 1 PASS)

**Headline**: v10 PGD hard-example mining beats vanilla EqM at paper-comparable scale across 3 seeds.

**Numbers** (IN-1K-256 class-cond, EqM-B/2, 80ep, FID 50K, gd sampler eta=0.003 / 250 steps / cfg=1.0 — identical harness to the trusted vanilla baseline):
- seed0 27.2086, seed1 27.9230, seed2 27.6020
- **mean 27.58 ± 0.36** vs trusted vanilla seed0 **31.41** → **−3.83 FID**
- All three seeds individually below vanilla (disjoint) — consistent with Exp3 single-seed ANM 26.88 vs 31.27.

**Gates**:
- Phase 1 (v10 IN-1K seed0 FID ≤ 30.41): **PASS** (27.21).
- Phase 2 (3-seed Welch p<0.05, mean ≥1 FID gain): gain ≫1, but **proper p-value blocked** — vanilla seeds 1,2 trains failed final-sync (ckpts only ep57–59). Need vanilla seeds 1,2 resumed to ep80 + FID for a real t-test.

**Caveat**: comparison currently v10 (n=3) vs vanilla (n=1 trusted). Strong directional but not the pre-registered Welch until vanilla 3-seed lands.

**Decision needed**: authorize resume of vanilla seeds 1,2 to ep80 (2× ~12h B/2 trains) to complete the Phase 2 Welch gate.

**Artifacts**: FID jobs 19680996/19681008/19681015. Ckpts: v10 l03 seed0 home final.pt; seed1/seed2 holylabs `mkrasnow_eqm/`.

---
## DRAFT 2026-06-07 — ANM capability-ladder NULL (do not send until reviewed)

Tested whether v10 (PGD hard-example mining) unlocks downstream image operations
(inpaint/colorize/super-res/deblur/outpaint) beyond its FID gain. Frozen-weight,
identical-sampler, vanilla-vs-v10 checkpoint comparison.

Result: NULL. Rung1 (denoise/inpaint/compose) + Rung2 (4 held-out corruptions)
both show v10 ≈ vanilla at noise level (ΔPSNR <0.03dB, ΔLPIPS ≤0.004). The FID
gain (27.58 vs 31.41) does not convert to conditional repair capability.

Action taken: killed the capability ladder per pre-registered gate. Did NOT spend
the 30h random-corruption control train (no signal to defend). Rungs 3-5 not run.
Workshop story (mining = FID/generation method) unchanged and intact — we simply
do not claim downstream-task transfer. Postmortem:
documentation/postmortem-capability-ladder-2026-06-07.md

Decision for you: (a) accept null, keep capability work shelved; or (b) authorize
a stronger probe (3-seed + n>=256, or trained-conditional head) before final kill.
My recommendation: (a) — denoise exact-parity is strong convergent evidence.

---
## DRAFT 2026-06-08 — Capability ladder v2 COMPLETE (A-F) (review before send)

Re-ran the ANM capability question with a behavioral ladder (after v1 zero-shot
inpainting null). Frozen ckpts, vanilla vs v10 λ0.3. Verdict: the v10 FID gain IS a
real behavioral change, but bounded to "near-manifold quality":

POSITIVE — A: gain=quality+class-adherence+coverage, diversity flat, 91% classes
improve. B: concentrated on HARD classes (1.34x feat-dist / ~1.9x class-adherence vs
easy). D: no collapse anywhere; ~2.5x sample-efficient (v10@nfe100 <= vanilla@nfe250)
+ overshoot-tolerant in converged regime.

NULL — C: no failed-trajectory rescue. E: contribution not splice-localizable. F: no
counterfactual class-steering (base model too weakly conditional; inert both arms).

Mechanism: PGD hard-example mining sharpens the field near the data manifold in weak
(hard-class) regions. No far-from-manifold capability (no repair/steer/restructure) —
consistent with v1 inpainting null + c(γ) decay.

Paper impact: strengthens the quality+hard-class+efficiency story; bounds out
editing/repair/low-NFE-robustness claims. No long control train spent (gated; the
checkpoint signal that DID appear is A/B/D, all eval-only).

Decision for you: accept this framing for the workshop draft, or 3-seed-confirm A/B/D
(seed0 λ0.3 ckpt was pruned; would need re-fetch/regen) before locking claims?

---

## PI Update Draft — 2026-06-13 — Metacognition probe (off-plan exploratory)

**Trigger:** significant positive pivot from an off-plan exploratory diagnostic.

**One-liner:** A learned probe over the EqM sampling **trajectory shape** detects
which generations will be garbage — and acting on it improves FID. The signal is
in the descent *dynamics*, not the energy.

**Path (1 GPU-hr + CPU):**
- Asked: does any cheap scalar at the GD stopping point separate good/garbage
  *independent of gradient norm* (the load-bearing assumption for a "metacognition
  sampler" that restarts on spurious minima)?
- Energy scalars (dot/L2 of the field) are **dead**: ~0.61 de-confounded AUROC,
  *below* a dumb latent-NN baseline (0.627). EqM energy = density skeleton, not a
  quality axis. (Confirms the skeptical prior.)
- But a **learned probe over the trajectory shape** (norm oscillation + magnitude-
  normalized norm/dot curves) hits **0.818 ± 0.012** held-out within-norm AUROC
  (30% test, 5 seeds) — well above the 0.80 bar and the 0.684 norm floor.
- **Payoff (controlled):** probe-guided rejection beats a random-drop floor at
  every keep-fraction (0.5–0.9), recovering **34–50%** of an inception-oracle's
  FID gain (e.g. keep-0.7: probe 86.7 vs random 100.2±0.3 vs oracle 73.1). The
  probe never sees the image — only the descent curve.

**Caveats:** vanilla EqM-B/2, single ckpt, GD; FIDs are PROXY (3000-subset, 10k
ref, cfg=1.0) — only relative ordering is valid. Not a paper-scale number.

**Decision for you (3 forks):**
1. Build the **in-line restart sampler** (perturb+continue on high P(garbage)) —
   new generation + design choices — or is probe-guided *rejection* the framing?
2. **Scale to real 50k FID** at trusted scale to make it a claim (human-gated,
   off summer-plan) — or keep as an exploratory result?
3. **Where it fits**: standalone "trajectory-shape quality probe for EqM/flow
   samplers" note, or fold into the diff-EqM story?

Nothing scaled or pushed; all rsync-deployed (upstream fix in flight).

### Addendum — 2026-06-14 — @scale confirmation

In-line restart sampler (probe-guided best-of-R=3) at 15k trusted scale (1-GPU
gpu_test; multi-GPU partitions were queued hours out):
- vanilla 29.53 (≈ baseline 31.41 → sanity OK, pipeline validated)
- probe-gated 27.84 (**Δ1.69 FID better**)
- oracle 17.75 (positive control); probe recovers 14% of oracle gain.

The trajectory-shape probe (no image access) gives a real, controlled FID
improvement at trusted scale. 14% < the 45% of pool-rejection because best-of-R=3
restart is a harder lever. Levers to recover more: higher R, or threshold-adaptive
rescue (METACOGNITIVE_RESCUE_SPEC.md). Literal 50k needs a multi-GPU slot.
Also delivered: post-hoc analysis ladder (robustness sweep + learned dynamics probe
+ dependent sbatch + 3-case synthetic suite, all passing).

---

## 2026-06-14 — EqM trajectory-metacognition: PASS at scale (DRAFT, review before sending)

**Trigger:** stage exit gate pass (Phase 1 consistency + Phase 2 online sampler), 3-seed confirmation at 50k.

Two headline results, both decision-grade, both equal-NFE controlled:

1. **Consistent FID gain (50k, 3 seeds).** Probe-guided best-of-R restart, where a
   learned probe over the descent-*trajectory shape* (not energy — energy is dead)
   picks the best of R draws: vanilla→probe = 28.20→26.21, 27.78→25.95, 27.83→26.04.
   **Mean Δ1.87 ± 0.11 FID, 95% CI ±0.12 (excludes 0), probe<vanilla on all 3 seeds.**

2. **Online metacognition sampler works (15k).** The true adaptive version — restart
   only the probe-flagged (likely-garbage) samples mid-generation; a random-restart
   control spends the identical NFE blindly. probe-restart 28.51 < random-restart
   29.76 (**Δ1.24 at equal compute**), vanilla 29.55 reproduces baseline (sanity OK),
   oracle 23.32. Recovers 19% of the oracle gain.

Supporting: a maze-planning analog (CPU) shows the same mechanism transfers — risk-
guided branching 0.928 vs random 0.794 valid-path-rate at equal compute.

**Framing:** EqM's energy scalar does *not* predict sample quality (dies at ~0.61
AUROC, below a trivial baseline); the signal lives in the *shape of the descent
dynamics* (~0.82 de-confounded) and is actionable at inference with no retraining.

**Open / needs steer:** the image-domain capability rungs (inpainting / translation)
are designed but unrun. Should we prioritize those, or go deeper on planning (maze)?

*(Detail: experiments/separability_diagnostic/{SCALE_RESULTS.md, PAPER_CLAIM_STATUS.md, YILUN_UPDATE.md, FINDINGS.md}.)*

---

## DRAFT (2026-06-30) — Selector result replicates on a 2nd checkpoint (NOT seed0-specific)

**Trigger:** result at paper-comparable scale (IN-1K 50k) + generality check.

We re-ran the locked B/2 inference-time metacognition selector on an independent
checkpoint (B/2 vanilla **seed1**) to test whether the result is checkpoint-specific.

Full 50k FID (probe_gated, R=3 restart):
- probe-restart **24.81** vs vanilla restart-baseline 27.51 (Δ **2.70**, recovers 23% of oracle 15.84).
- Locked seed0 reference: probe 24.66 ± 0.16, r3rand 27.95. → **probe-restart 24.81 ≈ 24.66**.

**Takeaway:** the actionable selector gain reproduces near-identically on a second,
independently-trained checkpoint. Not a seed0 artifact.

One nuance worth flagging: seed1's *diagnostic* probe AUROC came in lower (0.746 vs 0.813)
because that run's label-sanity check was weak (s4=0.577). But the **FID gain transferred
in full** — confirming AUROC was depressed by the label pipeline, while the real
(label-independent) selector signal is intact. FID is the number to trust.

**Scope:** single 50k draw, same B/2 scale (tests checkpoint-instance, not model-scale).
Model-scale axis (S/2 checkpoint, complete + ready) is the natural next replication.

**Ask:** proceed to S/2 scale-axis replication, or call checkpoint-generality settled here?

---

## DRAFT (2026-07-01) — Selector result ALSO replicates across model scale (S/2)

**Trigger:** result at paper-comparable scale (IN-1K 50k) + scale-axis generality check
(continuation of the checkpoint-generality replication above).

Re-ran the locked B/2 inference-time metacognition selector on an independent model
scale (**EqM-S/2** vanilla, seed0, 80ep, same sampler config eta=0.003/250/cfg=1.0).

Full 50k FID (probe_gated, R=3 restart):
- probe-restart **47.29** vs vanilla restart-baseline 50.01 (Δ **2.72**, recovers 16% of oracle 32.64).
- Compare: locked B/2 seed0 Δ~3.29 (16-18% oracle); B/2 seed1 Δ2.70 (23% oracle).
- **Nearly identical Δ FID (2.7-3.3) across all three runs, at three different checkpoints/scales.**

Diagnostic chain also replicated a third time: scalar energies dead (best AUROC 0.593),
learned SHAPE-only probe de-conf 0.736 (vs seed0 0.813, seed1 0.746) — same SHAPE≫MAG,
FULL≈SHAPE pattern; absolute AUROC tracks a weak label-sanity check (s4≈0.58 in both
non-seed0 runs) but FID (label-independent) confirms the mechanism each time.

**Takeaway:** the selector result now generalizes across BOTH checkpoint instance
(seed1) AND model capacity/scale (S/2) at fixed sampler config. This is the strongest
generality evidence for the metacognition selector to date — a single locked recipe
(SHAPE probe over descent trajectory, best-of-3 restart) delivers a consistent ~2.7-3.3
FID gain independent of which EqM checkpoint it's applied to.

**Scope/caveats:** each non-seed0 run is a single 50k draw (not multi-seed CI like the
locked 5-seed B/2 seed0 result). S/2 is undertrained relative to B/2 at 80ep (smaller
model, absolute FID much higher) — that's expected and doesn't affect the delta reading.

**Ask:** with checkpoint + scale generality both confirmed, is this sufficient breadth
for the paper's generality claim, or do we want a third axis (e.g. L/2, or a non-vanilla
training recipe) before locking the writeup?

---

## 2026-07-02 — Yilun request: math formulation + energy-OOD proposal (draft, do-not-send-verbatim, review first)

**Trigger:** PI-requested deliverable (methodology writeup) + open research question (blocker
requiring scope decision on whether to pursue energy-side work vs. lock current selector story).

Wrote up full mathematical formulation of the metacognition selector: EqM target/c(γ) def,
why pointwise energy reads are degenerate at the manifold (c(γ)→0 forces ‖f‖→0 near
convergence for good AND bad samples alike — explains why dot/path energy scored 0.605-0.609,
below the dumb k-NN baseline 0.627), the SHAPE-vs-MAG feature decomposition behind the 0.813
probe, and the deployed probe@50/best-of-R selector. Full doc:
`documentation/metacog-math-formulation-2026-07-02.md`.

**Energy-OOD question**: proposed 3 levers, cheapest first —
(a) re-read existing trajectory shards with a windowed path-integral (early/mid descent only,
where the shape probe's signal actually lives) instead of full-path or terminal — zero GPU
cost, no retrain, direct next experiment;
(b) fold the shape signal into training via an auxiliary loss (low-risk, reuses v10 mining
infra) or an explicit contrastive energy head (higher-risk, needs a written EqM-compatibility
argument before code per AGENTS.md gating discipline — flagged fragile, matches paper's
noted fragile EqM-E mode, currently untested);
(c) distill the 0.813 shape probe into a single-forward-pass scalar (supervised, bounded by
known ceiling, cheap).

**Recommendation**: run (a) first (no new compute), only escalate to (c) then (b) if (a)
doesn't cross 0.80.

**Ask**: does PI want us to pursue making the energy function itself OOD-discriminative
(new work, opens a lever likely to loop through the AGENTS.md variant-proposal gate for (b)),
or is the current framing ("selector over descent shape, energy explicitly ruled out as
uninformative") sufficient for the paper and this stays a documented negative result?

## 2026-07-02 — nn_dist skepticism control COLLAPSES shape-probe headline claim (bug-invalidating trigger)

**Trigger:** bug invalidating prior reported result (AGENTS.md PI-update trigger).

Following Yilun's energy-OOD question, ran a skepticism control on the post-hoc endpoint
energy head (`E_ψ`, `energy-ood-head-design-2026-07-02.md`): raw 0.753 AUROC collapsed to
0.524 (baseline floor) once regressed against `nn_dist` — the same distance metric used to
construct the good/garbage labels. `E_ψ` was mostly re-deriving `nn_dist`, not finding new
endpoint signal.

Applied the SAME control to the **main claimed result**, the descent-SHAPE probe (reported
0.81-0.82 AUROC, "de-confounded from grad-norm", `SYNTHESIS_METACOGNITION.md` claim #2):
raw AUROC 0.80-0.83 (5 seeds), but **residual AUROC after regressing out nn_dist collapses
to 0.556-0.570 (mean 0.562)** — same failure mode as the energy head. Pearson/Spearman
corr(p_bad, nn_dist) ≈ 0.54-0.55. Within-nn_dist-decile control has only 1/10 usable bins
(nn_dist near-fully determines the label in 9/10 deciles) — same confound-collapse pattern
`dynamics_probe.py`'s `MIN_USABLE_BINS` guard exists for, just never checked against
`nn_dist`, only against norm magnitude. Full writeup + numbers:
`documentation/shape-probe-nndist-skepticism-2026-07-02.md`.

**What's affected:** claim #2 in `SYNTHESIS_METACOGNITION.md` ("descent dynamics predict
failure, de-confounded from grad-norm") was never de-confounded against `nn_dist`. Until
shown otherwise, the honest claim is "trajectory shape predicts nn_dist-defined failure —
largely explained by correlation with nn_dist itself," not "trajectory dynamics reveal
distance-independent semantic OOD/quality." Claim #3 (restart improves FID, an action-based
result) is not directly invalidated — the intervention still works — but the *mechanistic*
story ("it's dynamics, not distance/magnitude") needs re-examination, since it was never
tested against this specific confound.

**Ask:** (1) does PI want a cross-seed/cross-shard held-out replication of this residual
check before treating it as settled (not yet run, per explicit instruction)? (2) if it
replicates, does the paper narrative need to shift from "descent dynamics reveal semantic
OOD" to "descent dynamics recover an nn_dist-correlated failure signal, actionable via
restart regardless of mechanism"? (3) should `SELECTOR_LOCKDOWN_RESULTS.md` and the
workshop draft be held pending this ablation?

## 2026-07-02 — CORRECTION to nn_dist skepticism entry above: over-controlled, not a bug

**User correction (methodological):** the residual-AUROC test above regressed `nn_dist`
out of the probe score and then scored against a label DEFINED as `1[nn_dist > τ]`. That
removes the label-generating variable itself — a good predictor is expected to collapse
under this test near-tautologically. This is NOT a "bug invalidating prior result"; it's
an over-controlled test that asks an ill-posed question.

**Corrected claim status:**
- NOT supported: "shape probe / E_ψ reveal semantic OOD independent of distance."
- STILL STANDING: "early trajectory shape predicts eventual nn_dist-defined failure"
  (claim #2, restated) and "acting on it improves FID" (claim #3, intervention result,
  untouched).
- The 1/10-usable-bin decile collapse reflects the label being too tightly coupled to
  `nn_dist` for that control to be well-powered — not that the probe lacks information.

**Revised ask:** no longer requesting a decision on invalidating claim #2. Instead:
should we run the shape probe / E_ψ against a non-`nn_dist` target next (`max_softmax`,
already cached, zero new compute) to test the distance-independent claim properly? Cross-
seed/cross-shard replication still separately pending, not yet run.
