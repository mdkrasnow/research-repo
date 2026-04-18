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

### 2026-04-18 — Publishability plan documented; Stage A in progress
**Trigger**: state-file setup (not yet sent to PI)
**Sent to PI**: no — first real update will be when Stage A exit gate produces data
**Summary**: Goal set to NeurIPS/ICML/ICLR. Autoresearch at proxy scale (2ep IN-100, FID 278→253, -9.2%) — not publishable as-is. Stage A exit gate = 3-seed repeatability check on best proxy config. Round 4 (9 candidates across 4 dimensions) running now.

## Reminders for the agent
- Draft updates in this file (under the log) **before** sending, so the PI gets a coherent story rather than reactive pings.
- When a triggered event fires, append a draft update entry within the same session and flag `needs_user_input.value=true` in pipeline.json with a prompt like "Draft PI update ready in documentation/pi-updates.md — review and send?"
- Never fabricate headline numbers. If an experiment is still running, say "in progress, expected by <date>."
- Weekly digest: the agent should prepare a draft Monday morning even if nothing triggered, so the PI has a consistent signal of life.
