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
