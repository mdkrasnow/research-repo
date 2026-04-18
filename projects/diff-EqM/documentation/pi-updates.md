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
