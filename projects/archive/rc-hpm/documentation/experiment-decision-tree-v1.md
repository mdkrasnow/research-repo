# RC-HPM × EqM Experiment Decision Tree — v1 (superseded by experiment-decision-tree.md v2)

Kept because v2 references nodes "as v1". v2 overrides on any conflict.

## Key branch variables
- K1 calibration validity — realized risk ≤ α on holdout (machinery works)
- K2 damage exists — uncontrolled mining CAN hurt (premise for risk control)
- K3 utility exists — mining HELPS in concentrated-error regime
- K4 risk control preserves utility while bounding damage (paper claim)
- K5 EqM compatibility — survives c(γ) geometry
- K6 scale transfer

```
LEGEND  [E#]=experiment  <G#>=pre-registered gate  ──▶ pass  ┄┄▶ fail
        (R)=single retune budget then kill   (K)=kill+postmortem
        ⊕/⊖ = positive/negative control arm (mandatory, same run)
        $=CPU min-hrs  $$=CPU hrs / 1 small GPU  $$$=GPU days

══════════ STAGE 0 — STATISTICAL MACHINERY (CPU, $) ══════════

 [E0.1] Synthetic contrastive toy (Gaussian mixture clusters, known classes,
        tiny MLP encoder). Full spec pipeline: gate → MINE_BATCH →
        CALIBRATE_LTT (HB p-values, fixed-seq path, D_fit/D_test split) →
        L_out loss → drift monitor.
        ⊕ oracle gate (q = true label)   ⊖ random gate
        <G0: K1 calibration valid?> (REWRITTEN in v2)
 [E0.2] Damage probe. Same toy + planted label noise CONCENTRATED near
        cluster boundaries. Arms: no-mine / naive hard-mine (no cert) / RC-HPM.
        ⊕ SupCon oracle    ⊖ no-mine baseline
        <G1: K2 damage + K3 utility?>
        both hold → STRONGEST premise, Stage 1.
        only K3 (no damage) → escalate noise ×1 (R) (REMOVED in v2: fixed ladder),
          still none → "certified = same gains + guarantee" framing, flag.
        only K2 (no utility) → abstention autopsy, α sweep ×1 (R)
          (REMOVED in v2: α not tunable, P2), still dead → (K) contrastive-RC;
          endpoint arm may live → E1.2.

══════════ STAGE 1 — EqM-LITE BRIDGE (CPU, $–$$) ══════════
 ARM A (disp-pair, spec-faithful):
 [E1.1] 2D EqM toy (paper Fig-1 setting, c(γ) trunc-decay EXACT as get_ct).
        EqM loss + RC-HPM dispersive term on penultimate activations.
        Metrics: field error vs reference, mode coverage, attractor purity.
        ⊕ oracle pairs  ⊖ vanilla EqM
 ARM B (endpoint-cert, v10 lineage):
 [E1.2] Same 2D toy. ANM PGD mining + gate certifies mined endpoint flows to
        correct basin under frozen teacher field; eps_ball CALIBRATED by LTT.
        ⊕ oracle basin check  ⊖ uncerted ANM eps×10 (damage)
        <G2: K5 EqM-compat, per arm?>
        (a) no collapse, loss finite
        (b) aux/base ratio stable, non-saturating (v02 lesson)
        (c) arm ≥ vanilla EqM on primary metric
        (d) ⊖ uncerted-aggressive arm visibly damages field (K2 at EqM scale)
        both pass → carry both | one passes → carry winner |
        (c) fails both but (d) holds → harm-bounding-only framing |
        (b) saturates → mechanism fights c(γ); (R) c(γ)-reweight or γ-window;
        fail again → (K) EqM bridge dead, contrastive standalone survives.
 [E1.3] MNIST EqM-mini. Winning arm(s) vs vanilla. MANDATORY smoke sample
        probe: ≥16 sampled images + tiny-FID ≤2K (CAFM postmortem rule).
        <G3: tiny-FID(arm) ≤ tiny-FID(vanilla)+ε AND visual sanity AND
        realized risk ≤ α throughout AND abstention < 50%>
        fail → (R) one k/nuisance retune → fail → (K) + postmortem.

══════════ STAGE 2 — GPU CIFAR VARIANT HARNESS ($$, days) ══════════
 [E2.1] CIFAR 150ep, arms: v00 / ANM-v10 / RC-HPM winner / ⊕ oracle-gate /
        ⊖ corrupted-gate (spec E4). Diagnostics every 200 steps.
        <G4> (a) RC ≥ v00 (b) corrupted-gate: abstention↑, risk ≤ α
        (c) wall-time ≤ 1.5× vanilla.
        RC ≥ v10 → STRONGEST | RC ≈ v10 → "same gains, certified" |
        RC < v10 by >1 FID → validity-power tradeoff (spec E9), report.
        b fails → guarantee broken, no scale-up until fixed.
        a fails → (R)×1 → narrow to harm-bounding or (K).

══════════ STAGE 3 — IN-1K ($$$; gate-locked, human approval) ══════════
 [E3.1] EqM-B/2 IN-1K 80ep seed 0 vs vanilla 31.41 / v10 27.58.
        <G5: FID ≤ 30.41 AND realized-risk table ≤ α> (ROLE CHANGED in v2:
        continue/stop only)
 [E3.2] 3 seeds → claim gate. Ablations (spec E8) at CIFAR scale parallel.
```

Paper outcomes that survive partial failure: (1) certified AND better;
(2) same gains + guarantee; (3) harm-bounding only. Zero-paper only if
machinery itself broken.
