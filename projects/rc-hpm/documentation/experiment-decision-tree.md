# RC-HPM × EqM Experiment Decision Tree — v2 (adversarially patched)

Inherits v1's structure (pre-registered gates, cheap→expensive, ⊕/⊖ controls,
1-retune budgets, kill+postmortem). This file lists the global protocol fixes,
then the revised tree. Changes vs v1 are tagged [Fn] (fix n).

================================================================================
## GLOBAL PROTOCOL (pre-registered before any run)
================================================================================

P1 [F2, REWRITTEN in v2.1 — v2's "1%" was ARITHMETICALLY IMPOSSIBLE]:
    Batch-level exchangeability requires calibration batches at the TRAINING
    batch size n, and disjoint batches for i.i.d. losses. The budget is
    therefore DERIVED, not declared:
        N_lab = M·m_per_fold·n  +  |D_gate|  +  |D_holdout|
    Worked: n=256, m=50 batches/fold, M=4 ⇒ 51,200 calibration examples —
    MORE THAN ALL OF CIFAR. The slogan budget ("1%") and the spec's batch-level
    calibration cannot coexist at CIFAR scale. Pre-registered resolution:
      (a) reduce the TRAINING batch size for all arms to n=64 (a pipeline
          choice, legal if every arm shares it), m=20, M=4 ⇒ 5,120 + gate
          2,000 + holdout 1,000 ≈ 8,200 labels ≈ 16% of CIFAR — report this
          honestly; at IN-1K the same absolute budget is ≈0.6%, where the
          semi-supervised story actually lives; AND/OR
      (b) empirical-pool fallback where (a) is too costly: sample calibration
          batches with replacement from the labeled pool; the LTT guarantee
          then holds w.r.t. the POOL's empirical distribution, and the
          pool→population gap is reported as an explicit second-level error
          (DKW-style slack in N_cal), not hidden.
    γ-conditioning (P4) fragments this budget further per bin — the per-bin
    arithmetic must be checked before E1.1, not discovered during it.
    Every labeled-data baseline (SupCon-on-budget, semi-supervised arm) gets
    the SAME derived budget. Full-label SupCon runs only as the ⊕ ceiling.

P2 [F7] α IS NOT TUNABLE: pre-register the grid α ∈ {α₀, 2α₀, 4α₀} as
    primary/secondary/tertiary endpoints NOW. v2.1 compute scoping: the full
    grid runs at Stages 0–1 (cheap); at Stage 2+ only α₀ runs on every arm,
    and the grid runs on the single winner arm — otherwise the grid triples
    GPU cost and the "$$, days" label is fiction. No gate may "retune α";
    retune budgets apply only to nuisance knobs (k, τ_w, layer choice,
    c(γ)-reweight). Any λ retune at any stage consumes a fresh calibration
    fold and is logged against δ = M·δ_r.

P3 [F9] GATE STATISTICS: every quantitative gate criterion must name
    (a) its primary metric (ONE — no OR-of-metrics; secondary metrics are
        descriptive only),
    (b) a margin pre-registered ABOVE the measured noise floor (bootstrap or
        seed-resample of the relevant metric on the vanilla arm), and
    (c) v2.1: a TREE-LEVEL false-kill budget. Per-gate false-kill of 10%
        compounds across the ~6 noise-sensitive gates to 1−0.9⁶ ≈ 47% — a
        true-good method would die in this tree roughly half the time.
        Size per-gate false-kill ≤ 3% (more seeds at cheap stages, wider
        margins only where extra seeds are unaffordable) for ≈ 84% tree-level
        survival of a true method, and STATE the survival number in the paper.
    (d) v2.1: a post-retune pass is a SECOND draw at the same gate — flag all
        post-retune passes as secondary-confidence results in reporting.
    CIFAR arms: 5 seeds (not 3). IN-1K: report effect sizes + all seeds; no
    p-value theater with n=3.

P4 [F5] γ-STRATIFICATION (EqM-specific A1): the gate A_ψ and ρ̂ are
    γ-conditional (trained/fit with γ-bin features or per-bin heads);
    calibration batches replicate the EXACT training γ-sampling law;
    the drift monitor runs per γ-bin (Bonferroni across bins AND checkpoints).
    Any change to the γ schedule = pipeline change = unconditional recalibration.

P5 [F4] ENDPOINT ARM LEGALITY: ANM/PGD mining runs ONLY against the frozen
    teacher field (EMA snapshot at calibration time), with pre-registered step
    count/size. Calibration units = labeled examples pushed through the
    IDENTICAL frozen PGD pipeline. Mining against the live student is forbidden
    in certified arms (it appears only in the ⊖ damage arm, uncertified by
    design).
    v2.1 — BASIN-LABEL FUNCTION must be pre-registered, because at scale it is
    itself a model output, and LTT assumes calibration labels are ground truth:
      2D toy: analytic basins (exact) — this validates the procedure.
      Image scale: descend the frozen teacher field from the mined endpoint to
      convergence; label the attractor by a fixed pre-registered classifier;
      "correct basin" ⇔ attractor class == source-example class.
      The labeler's own error rate η is MEASURED (on labeled pairs where truth
      is known) and the reported guarantee becomes risk ≤ α + η — η is part of
      the published table, not a footnote. If η is not small relative to α,
      arm B's certification claim is not available at that scale.
    Wall-time of basin labeling (full descent per calibration endpoint) is
    charged to arm B's per-arm budget at G4(c).

P6 [F10] BASELINE ENVIRONMENT: vanilla EqM seed-0 is RE-RUN once in the rc-hpm
    environment at each scale before its number is used in any gate. Cited
    numbers (31.41, 27.58) are sanity references only, never gate inputs,
    unless env hash (commit + torch + data pipeline) matches the original runs.

P7 [F11] ABORT SEMANTICS: if CALIBRATE_LTT finds no valid λ, the arm trains
    un-mined and the event is a PASS for risk gates (vacuous control is the
    honest behavior) and a recorded DATA POINT for utility gates (throughput 0).
    ⊖ random-gate arms are expected to land here or at near-total abstention.

================================================================================
## REVISED TREE
================================================================================

LEGEND as v1. New/changed nodes marked ★.

═════ STAGE 0 — STATISTICAL MACHINERY (CPU, $) ═════

★[E0.0] [F12] BUG-INJECTION SUITE (entry condition for everything).
    Unit tests on synthetic data where ground-truth risk is computable.
    Each deliberately re-introduced bug must push realized risk > α, and the
    clean implementation must keep it ≤ α:
      fold reuse (ρ̂ fit on D_test) / asymmetric teacher pretraining (A1′) /
      partitioned batches + Bentkus / undefined w′ path / zero-budget batches /
      monitor without Bonferroni / live-student mining in endpoint pipeline /
      ★v2.1: γ-pooled gate where the truth is γ-conditional (P4 violation) /
      ★v2.1: overlapping calibration batches from an undersized labeled pool
      treated as i.i.d. (the P1 failure mode — must inflate realized risk or
      be caught by the dependence check).
        <G-1: all injections detected, clean run passes> pass → E0.1
        fail ┄┄▶ fix tests/impl; no other node may run. (No retune budget —
        this is correctness, not science.)

 [E0.1] Synthetic contrastive toy, full pipeline, 20 seeds,
        ⊕ oracle gate  ⊖ random gate.
   ★<G0 (rewritten) [F1] + v2.1: K1 calibration valid?>
        Pre-register δ (e.g. 0.1). PASS iff:
        (a) #seeds with [empirical holdout mean − 2·SE] > α  is ≤ binomial
            upper bound Bin(20, δ) at level 0.05  — i.e. test the GUARANTEE
            P(E[L]≤α) ≥ 1−δ, not a folk "95% of seeds" rule that a correct
            implementation fails ~60% of the time at δ=0.1; and
        ★(a′) v2.1 POWER FIX: (a) alone is nearly blind to SYSTEMATIC mild
            miscalibration (e.g. E[L] ≡ 1.2α rarely trips a per-seed CI).
            Add the pooled test: the across-seed mean of holdout means, with
            its CI, must be consistent with the δ-mixture implied by the
            guarantee (pooled mean ≤ α + tolerance derived from δ and the
            per-seed exceedance cap). Catches the bug class where every seed
            is slightly, consistently wrong; and
        (b) ⊖ random gate: per P7, abstention →1 or ABORT, with realized risk
            ≤ α whenever any mining occurred.
        pass → E0.2     fail ┄┄▶ statistics bug; fix impl (E0.0 suite first),
                              re-run. Thresholds untouched.

★[E0.2′] [F3] adds two arms to v1's E0.2: RINCE (two-line loss swap) and
        τ-sweep over the SAME toy; plus [F15] "certified-random-k" arm
        (certification without hardness) to decompose mining vs certification.
        Noise levels for the planted-concentration probe are pre-registered
        as a fixed ladder {low, med, high}; ALL levels run; no escalation
        retune. [F6 partial]
   <G1 (amended)>: as v1, PLUS:
        (e) RC-HPM > RINCE at the concentrated-noise level by the
            pre-registered margin, at matched compute. If RINCE ≥ RC-HPM
            everywhere ┄┄▶ (K) for the gains claim at toy scale; the
            harm-bounding + guarantee framing survives only if E4-style
            corruption shows RINCE silently degrades where RC abstains.
        (f) certified-random-k vs RC-HPM separates "certification helps"
            from "hardness helps" — recorded, informs Stage-1 framing.

★[E1.0] [F6] TARGET-DOMAIN PREMISE (CPU, $; THE spec-E1 node, was missing).
        Precompute EMA embeddings on CIFAR train; with labels, estimate ρ(s)
        and the gradient-weighted damage curve w(s)·ρ(s); same per γ-bin
        on noised inputs (P4 preview).
   ★<G1.5 (v2.1 precision fix): premise in the wild?>
        Primary, stated in DENSITY not mass — v2's "top-decile mass ≥ 3×
        bottom-half mass" mixed bin widths (10% vs 50% of pairs), silently
        demanding 15× density enrichment: under uniform errors the ratio is
        0.2, so "≥3" was 15× — an accidental near-impossible gate.
        Corrected: average damage DENSITY (damage per pair) in the top
        hardness decile ≥ 3× the average density in the bottom half.
        pass → Stage 1 with the motivation figure in hand.
        fail (flat ρ̂) ┄┄▶ the ENTIRE gains premise is weak on this domain:
        proceed ONLY on the harm-bounding/guarantee framing, with the flat
        curve reported. (No retune — you can't retune reality.)

═════ STAGE 1 — EqM-LITE BRIDGE (CPU, $–$$) ═════

 ARM A (disp-pair):  [E1.1] as v1, with P4 γ-conditioning, PLUS
   ★[F13] third sub-arm: RC-repulsive-only (certify negatives, keep the
   dispersive/repulsive-only form; no attraction term). Isolates
   "certification effect" from "mechanism swap to InfoNCE" — without it, a
   G2 failure can't distinguish 'certification hurts' from 'attraction
   fights the regression objective'.
 ARM B (endpoint):   [E1.2] as v1, under P5 (frozen-field PGD only;
   ⊖ live-student eps×10 arm is the damage control).

   <G2> as v1 but [F9]: criterion (c) names ONE primary metric per arm
   (pre-register: field error for A, attractor purity for B); the other
   becomes descriptive. (a),(b),(d) unchanged. c(γ)-compat retune as v1
   (it's a nuisance knob, allowed under P2).

 [E1.3] MNIST EqM-mini, as v1.
   <G3 (amended)> [F9]: ε in "tiny-FID(arm) ≤ tiny-FID(vanilla)+ε" is set to
   2× the bootstrap SD of tiny-FID measured on the vanilla arm (resampled
   2K-subsets) — pre-registered numerically before the run, not after.
   Risk monitor + abstention < 50% as v1.

═════ STAGE 2 — GPU CIFAR ($$, days) ═════

★[E2.1′] arms (5 seeds each, [F9]):
     v00 vanilla (RE-RUN, P6) / ANM-v10 uncerted / RC-HPM winner /
     ⊕ oracle-gate RC / ⊖ corrupted-gate RC /
     ★ naive-aggressive-mine no-cert [F8 — K2 damage evidence AT SCALE,
       was toy-only] /
     ★ RINCE at matched compute [F3] /
     ★ SupCon-on-budget + semi-supervised baseline on the SAME 1% labels [F2]
   <G4 (amended)>
     (a) RC-HPM ≥ v00 by pre-registered margin > seed noise floor (5 seeds)
     (b) corrupted-gate: abstention↑, realized risk ≤ α   (unchanged; the
         load-bearing certification demo)
     ★(b′) naive-aggressive arm damages FID or representation probes at
         CIFAR — K2 at scale. If (b′) fails AND RINCE ≈ RC: the method has
         no demonstrated problem to solve at this scale ┄┄▶ harm-bounding
         workshop framing or (K). Decided here, before any $$$.
     (c) wall-time ≤ 1.5× vanilla, PER ARM (endpoint basin certification
         costs descent trajectories — costed separately, not pooled) [F14]
   Branching as v1 (RC vs v10 trichotomy), with P2: no α retune branch;
   the α-grid endpoints were already run and are simply read out.

═════ STAGE 3 — IN-1K ($$$; gate-locked, human approval) ═════

 [E3.1] as v1 + P6 (vanilla seed-0 re-run in-env before G5 is evaluated).
   <G5 (v2.1 role change)> v2 treated G5's single-seed ≥1-FID margin as if it
   were evidence, but no IN-1K seed-noise floor exists or is affordable to
   measure — a 1-FID single-seed margin is unjustifiable as a CLAIM.
   G5 is therefore a CONTINUE/STOP gate only (does seed-0 look alive enough
   to spend 2 more seeds?); the CLAIM gate is E3.2's 3-seed mean. If internal
   diff-EqM history provides seed-variance data at this scale, cite it to
   justify the margin; else say so. λ-retune branch consumes calibration
   fold M and is logged against the δ budget [P2]; if folds exhausted,
   retune is FORBIDDEN — straight to workshop-paper path.
 [E3.2] [F9]: 3 seeds reported with effect sizes and per-seed table;
   the pre-registered claim gate is mean improvement ≥ 1.0 FID with all
   seeds individually ≥ 0 improvement, NOT a p<0.05 ritual at n=3.
   Ablations as v1.

================================================================================
## WHY THESE PATCHES ARE LOAD-BEARING (one line each)
================================================================================
F1  G0 previously killed correct implementations ~60% of the time (binomial
    math vs the actual (α,δ) guarantee).
F2  Without a label budget, SupCon-with-all-labels ends the paper in one row.
F3  RINCE is the cheapest falsifier of the gains claim; omitting it defers the
    deadliest test to the most expensive stage.
F4  Live-student PGD = θ-dependent pipeline = guarantee void at step one;
    frozen-field PGD restores exchangeability for arm B.
F5  Pooling γ levels makes gate scores a mixture; calibration silently breaks
    on any γ-schedule change.
F6  The premise (errors concentrate at the hard tail) was never measured on
    the target domain before GPU spend; planted toy noise can't establish it.
F7  Post-hoc α relaxation tunes the guarantee knob — the forking path the
    tree exists to kill.
F8  K2 damage evidence existed only at toy scale; the paper needs it at CIFAR.
F9  Un-margined point-estimate gates at 3 seeds are coin flips dressed as
    decisions.
F10 Cross-environment baseline citation at the headline gate invites a
    confound that one cheap re-run removes.
F11 An LTT ABORT is correct behavior; an undefined ⊖ pass condition lets a
    reviewer (or you) call honesty a failure.
F12 The bug-injection suite is the only test that verifies the machinery
    FAILS when it should — happy-path validation can't.
F13 Swapping repulsive-only dispersion for InfoNCE confounds certification
    with mechanism change; the repulsive-only RC arm decouples them.
F14 Per-arm wall-time stops the cheap arm subsidizing the expensive one
    through a pooled budget gate.
F15 Certified-random-k decomposes the two claims ("certification helps" vs
    "hardness helps") that the paper must defend separately.
