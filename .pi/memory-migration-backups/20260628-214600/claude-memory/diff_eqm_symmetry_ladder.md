---
name: diff_eqm_symmetry_ladder
description: Toy ladder testing symmetry-discovery vs hard-negative mining for EqM generalization; rungs 1-4 results and the blocker
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f4c5428-98a7-4292-b0a5-e56722645048
---

Exploratory toy ladder probing whether EqM should pursue symmetry-discovery over hard-negative mining
(v10). All CPU-seconds toys; filters, NOT publishable. NOW ITS OWN PROJECT: `projects/symmetry-discovery/`
(README + documentation/findings.md + experiments/latent_symmetry_rung{4..9}.py + results/ + pipeline.json).
Mirrored from diff-EqM symmetry_toys (still also there). Registered in projects/PROJECTS.md.

Results:
- Rung 1 (arithmetic, classification): equivariance CONSTRAINT generalizes perfectly to held-out;
  hard-negative mining = dead (~0). Same symmetry info as negatives vs constraint -> opposite outcome.
- Rung 2 (2D ring, symmetry linear in observed space): known + DISCOVERED rotation both fill held-out;
  discovery ties known once controlled. Hard-neg dead.
- Rung 3 (symmetry hidden by nonlinear decoder, free observed-space operator T): FAILS — T drifts
  off-manifold.
- Rung 4 (enc-linear-dec, T=dec(M.enc(x)), recon-frozen latent): FAILS, GENUINE negative. Recon good
  (0.034) but M collapses to identity. Latent-geometry diagnostic: recon latent is a DISTORTED loop
  (radii 2.6-9.4, angular gaps != 45deg) -> rotation is nonlinear there -> no linear M works.
  Reconstruction ALONE does not shape latent to linearize the symmetry.

Core theme: discovery works iff the operator family is expressible in the coordinates you search.
Hard-negative mining (v10's mechanism) installed NO manifold structure in ANY toy.

**Why:** for installing structure/generalization, symmetry CONSTRAINTS beat negative MINING; but
discovering the linearizing latent for a nonlinearly-hidden symmetry is the open blocker (chicken-egg:
latent must be shaped by the symmetry, recon alone won't do it).

LADDER rungs 1-9 (2026-06-04). Rungs 5-9 escalating diagnosis of unsupervised-discovery failure:
- 5: ORACLE_LATENT control (enc supervised to true z) -> recipe collapses M to identity even with a
  PERFECT clean latent. So blocker is the discovery objective, not latent geometry.
- 6: hinge anti-identity insufficient. 7: continuous manifold -> cyclic term pulls M to identity.
- 8: even hard-EXCLUDING identity (project ||M-I||=fixed) fails -> M drifts to an arbitrary non-symmetry
  matrix; arc unfilled. ROOT CAUSE: closure-via-field (eqm(f,T(x))) cannot anchor the operator to the
  data manifold because the FIELD CO-ADAPTS to whatever T outputs.
- 9: FROZEN DATA ANCHOR (energy-distance to unlabeled full-manifold samples, learn T before EqM, freeze)
  -> PARTIAL WIN. Fixes co-adaptation: T_onman 0.00 (field-closure) -> 0.70 (frozen anchor). BUT
  distributional anchor under-constrains the operator -> random on-manifold shuffle (shift_std ~95deg),
  arc NOT filled (recall 0.009 vs ORACLE 0.068); FROZEN~=VISIBLE (no symmetry actually learned). Frozen
  anchor NECESSARY but NOT SUFFICIENT; missing piece = operator COHERENCE (consistent flow). Rung 10
  (open, not run): frozen anchor + Lie-generator flow exp(t*xi) OR shift-inconsistency penalty.

- 10: frozen anchor + COHERENT operator (single latent matrix M instead of residual MLP) -> PARTIAL
  SUCCESS, best yet. Coherence fixed (shift_std 95->25deg), recovers a real rotation (M_angle ~-39deg),
  recall 0.025 (12x floor, ~37% of ORACLE 0.068).
- 11: orbit augmentation {M^k} to close coverage -> coverage hypothesis REJECTED (orbit==single, 0.024).
  Gap to ORACLE is operator PRECISION (discovered rotation fuzzy/seed-variable, shift_std 25, M_off 1.04),
  not coverage.

- 12: GROUP-STRUCTURED generator M=exp(theta*A) (generator A + step theta learned, direction NOT given)
  -> SUCCESS, NEAR-ORACLE. Fixes operator precision: shift_std 25->0.8deg(skew)/7.6(free); M_det
  0.56->1.00(skew)/0.90(free); recall 0.024->0.071(SKEW = ORACLE 0.067, exact rotation ~3deg/seed) /
  0.046(FREE weak-prior, ~69% ORACLE, precision partially emerges without rotation prior).

**REVISED FINAL VERDICT (rungs 1-12):** hard-neg mining (v10) installs NO manifold structure -- dead.
KNOWN-symmetry constraint = clean (ORACLE). UNSUPERVISED latent-symmetry discovery is VIABLE / NEAR-ORACLE
(NOT dead -- prior rung-8/11 'deprioritize' read was too harsh). 4 blockers all addressed: identity
(move) / co-adaptation (FROZEN DATA ANCHOR, not live field -- key insight) / coherence (single op) /
precision (GROUP GENERATOR exp(theta*A)). SKEW(rotation-family prior) hits ORACLE; FREE(general+det/ortho
reg, weak prior) ~69%. The gap was operator PRECISION, not impossibility.
- 13 (prior-budget): GEN_FREE_STABLE (det~1 + WELL-CONDITIONED cond->1, NO skew, NO rotation naming) =
  99% ORACLE, matches SKEW (det 1.00, shift_std 0.84, angle within 2.8deg). Rotation EMERGES from a
  STABILITY prior; you don't need to name it. Active ingredient = condition-number penalty (FREE without
  it stalls 69%). CAVEAT: 2D conflates stable~=rotation (det1+cond1=>orthogonal=>rotation); definitive
  prior-budget test needs higher-dim latent where stable strictly contains rotation (rung14).

- 14 (higher-dim, K=3 cylinder where stable!=rotation): GEN_FREE_STABLE (general 3x3, det~1+cond->1, NO
  skew, NO plane told) = 113% ORACLE, MATCHES skew arms. Discovered rotation CONFINED to active plane
  (passive_leak 0.04~=0) + h PRESERVED (passive_dh = plane-told ref) + exact group (det/cond 1.00). 2D
  conflation caveat DISSOLVED -- stability prior recovers the RIGHT symmetry + active subspace, no naming.

STRONGEST-FORM CLAIM: near-oracle unsupervised latent-symmetry discovery needs only: frozen data anchor
(co-adaptation) + coherent operator + group-structured generator exp(A) + STABILITY-ONLY reg (det~1,
cond->1). NO symmetry-class, plane, or subspace naming -- symmetry + active subspace EMERGE. Holds through
K=3 where stable strictly broader than rotation. Hard-neg mining dead throughout. Scope: still CPU toys
(clean enc->z latent, single abelian symmetry, energy-distance anchor); NOT yet shown at learned latent /
non-abelian / real high-dim EqM. For diff-EqM: strong enough to justify a non-toy test. Open: rung15
learned latent, rung16 non-abelian group, then real-EqM scale.

**How to apply:** (1) Prefer KNOWN/cheap symmetries injected as equivariance constraints/aug over more
hard-neg mining; for IN-1K, crops/flips/color are known -> inject directly, skip discovery. (2) Paper
framing: "symmetry constraints > mining for generalization WHEN symmetry is specified" + report
unsupervised discovery as analyzed negative. Do NOT claim unsupervised latent-symmetry discovery.
(3) Methodology: verify AE reconstructs BEFORE judging a symmetry result; always ship positive+negative
controls (ORACLE/ORACLE_LATENT were what cracked each diagnosis). All in doc + CLAUDE.md control rule.
See [[diff_eqm_phase_0_v10_pass]].

## CORRECTION 2026-06-04 — EqM bridge proxy PASSES; two prior "negatives" were bugs.
The spatial-proxy "targeting is the open problem / anchor-matching is direction-agnostic / discovered
-23deg wrong direction" conclusion (commits a0f5450, 9498f87) was a BUG ARTIFACT, now superseded.
Two load-bearing bugs in feature_gap_proxy_spatial_operator.py AND diff-EqM _stable_operator.py:
(1) the frozen encoder's forward/features were @torch.no_grad -> for spatial (image-space) operators the
anchor/energy-distance term contributed ZERO gradient to the operator A; "discovery" was driven only by
move+stability, so direction was the init seed and magnitude was the move target. Verified via autograd
(anchor term "does not require grad"). (2) move_target = pixel-dist of a 20deg rotation = held-band
center -> leaked the held magnitude (violates CLAUDE.md "don't force move = held angle").
FIXES: grad-flowing frozen encoder (feat_grad/features_grad: params frozen, NO no_grad, grad flows to
input); generic non-leaking move HINGE [5deg,45deg] (non-collapse+non-blowup only); and the real
positive lever -- MIXTURE-anchor objective: match (1-pi)P_real + pi*T(P_real) to fresh real, NOT
T(P_real) alone. CORRECTED proxy (quick+full agree, results_spatial_proxy_v2.json): MIXTURE_SINGLE
gap_HO 0.305 (+29.5deg, det0.99 cond1.13) << base 1.065, near KNOWN 0.182, beats random 1.41. T-only
UNDERSHOOTS (+4.9deg, sits at move floor) because visible mass dominates the anchor so a tiny rotation
already matches the bulk; the mixture lets (1-pi)vis cover the bulk and frees T(vis) to FILL the missing
held region -> discovers correct direction+magnitude UNSUPERVISED (held band enters only as unlabeled
anchor mass). KEY LESSONS: (a) anchor-matching is NOT inherently direction-agnostic -- the
mass-mismatched T-only objective was; MIXTURE generalizes T-only to ONE-SIDED gaps (toy rungs 12-14
worked w/ T-only only because their held-out was SYMMETRIC under the group). (b) Once gradient flows,
direction is DETERMINED (inverse worse) -> the "gauge/orbit" fix was itself a symptom of the no-grad
bug, not the cure. (c) ALWAYS autograd-check that a frozen feature extractor used INSIDE a discovery
loss is not @no_grad -- silent zero-gradient looks like "the objective doesn't work."
v12 UPDATED to validated recipe (mixture-anchor + grad features + move hinge, returns generator A;
EqM augments with GROUP exp(t*A) t~U(-1,1), aug_mode knob). CPU smoke passes (anchor 9.85->3.70).
Verdict now: mechanism passes the fast proxy honestly -> warrants ONE LONG CIFAR/FID run (40ep was
noise-limited). KNOWN-aug still the safe paper lever if FID disagrees. Commit 9061ceb.

## CIFAR-FID BRIDGE OUTCOME 2026-06-05 — NEGATIVE (kill v12->FID; proxy != FID predictor).
Ran the corrected v12 (mixture-anchor discovery + grad-flow + orbit aug) on real CIFAR EqM, 150ep, 1 seed,
FID(5000): v00 BASE 14.31; v12_random 13.63 (negative control BEAT everything); v12_discovered_orbit 14.41
(treatment ~=base, +0.10 NO benefit); vK_known rotate15 14.82 (positive control WORSE than base, FAILED).
Ordering random>base>discovered>known is wrong for the mechanism. KEY: the positive control (known rotation
aug) FAILED to beat base -> rotation is NOT a useful symmetry of CIFAR (canonical object orientation), so
augmenting with rotations (known or discovered) trains on off-distribution images and doesn't help FID;
v12_random won only as a weak generic regularizer. The fast proxy and FID DISAGREE on the same lever: proxy's
positive control (KNOWN rotation) strongly beat base because the proxy INJECTED rotation as ground-truth and
scored held-out rotation recovery; FID scores full-distribution generation where rotation aug is useless/harmful.
So the proxy validated the discovery MECHANISM (recover an injected symmetry) but the injected symmetry isn't
FID-relevant -> a feature-space held-out-gap proxy is NOT a valid FID predictor when the target symmetry isn't
exploitable by the generative task. DECISION: kill v12->CIFAR-FID; do NOT run more seeds (positive-control
failure means the lever itself doesn't help, multi-seed can't rescue). Discovery method still validated in
toys+proxy but needs a setting where the discovered symmetry genuinely helps the generative target. Confirms
prior "v12 mechanism fails on real CIFAR; CIFAR/FID noise-limited." Commit 81f6970.

## v13 SE2 BRIDGE RESULT 2026-06-05 — right family, WRONG PARADIGM (frozen single op lacks diversity).
After v12 (rotation) killed only the rotation ARCH, built v13: SE(2) 3x3 homogeneous affine (WITH
translation/crop) + CIFAR-appropriate positive control. Fast SE2 proxy PASSED (KNOWN_TRANSLATE_CROP
0.20<<base 0.70; DISCOVERED_SE2_MIXTURE 0.65<base,<random, learns near-pure translation). Full 150ep
CIFAR FID(5k), 1 seed: vK_translate_crop 12.59 (-1.72 vs base, WIN), v13_discovered 14.02, v13_random
14.08, v00_base 14.31, vK_rotation 14.82. v10_hardneg PENDING (job 19405301, fill in later).
READ: (A) HARNESS VALID — known random-crop is a STRONG CIFAR lever (-1.72), architecture critique
CONFIRMED (translation works where rotation hurt). (B) BUT v13_discovered ~= v13_random ~= base
(disc-random -0.06, disc-base -0.29 = single-seed noise) -> gain NOT from discovery.
FAILURE MODE (operator is EXCELLENT, paradigm is wrong): discovered op = clean ~2px diagonal translation
(tx-2.1 ty-2.0 det0.976 cond1.009 lin_off0.019, anchor 6.29->3.67 halved, aug ratio 0.88 stable). By
static metrics discovery succeeded. But (1) crop aug's VALUE IS DIVERSITY (different 2D shift each step,
81 pad4 crops); a frozen op gives ONE direction, orbit exp(tA) is a 1-param sliver -> low diversity. (2)
feature_shift_consistency ~0.009 (~0): translation is not a global feature shift through a conv encoder ->
op matches anchor DISTRIBUTION without a COHERENT transform the generator can exploit (rung-15 confound
again). (3) proxy passed b/c it scored "fill a STATIC held-out band" (point-op distribution MATCH) while
FID needs generative COVERAGE (a MEASURE of transforms). KEY LESSON: discovery-as-FROZEN-SINGLE-OPERATOR
does NOT transfer to augmentation-for-generation; aug value = a DISTRIBUTION of transforms, not THE one
operator. Fix would be "discover an aug distribution/subgroup + sample densely" but that -> "just do random
crop," questioning discovery's value-add over known aug for CIFAR. v12 vs v13: v12 wrong FAMILY (rotation),
v13 right family wrong PARADIGM. Commit 9bc29da. Bridge direction concluded; pivoting to a new CPU ladder.

## v14 CPU LADDER 2026-06-05 — v14 NOT authorized; discovery superfluous for known/generic CIFAR symmetry.
Tested whether discovering an aug DISTRIBUTION (vs v13 single frozen op) adds value. CPU-first, controls each rung.
A (anchor-grad gate) PASS: grad to operator 924.7, encoder frozen, no-grad path=0 (v12 bug repro).
B (move/leakage) PASS: broad hinge anchor-driven no-leak; KEY single-op UNDERDETERMINED on translation-spread
  anchor (no-guard drifts) -> motivates distribution. C (single vs distribution, translation-space coverage) PASS:
  single op CANNOT cover a 2D crop region (cover 5.19 vs base 1.40); discovered DISTRIBUTION covers it (0.32 vs
  oracle 0.017), high-rank 2D support (eig_ratio 0.39) emerges WITHOUT entropy floor. D (aug-training small CNN,
  3 seeds) DECISIVE NEGATIVE: translated-val acc base 0.391, known_crop 0.431, single 0.377 (v13 single HURTS),
  random_dist 0.421, disc_dist 0.420 -> discovered distribution == random distribution (closes 72% gap, crushes
  single, but adds NOTHING over random). E (EqM-lite) INCONCLUSIVE (translated-field gap 0.0014 within noise;
  velocity field eps-x inherently translation-robust; killed full run as wasteful).
VERDICT: v14 NOT authorized for CIFAR/FID. The distribution mechanism FIXES v13's under-diversity (A-C) but the
DISCOVERED distribution equals a RANDOM one in value (D). ARC SYNTHESIS: v12 wrong family (rotation), v13 right
family wrong paradigm (single op), v14 right family+distribution but discovered==random. => for KNOWN+GENERIC
nuisance symmetries on CIFAR (translation/crop) unsupervised DISCOVERY is superfluous; the useful aug is just
random crop (known control wins). Frozen-anchor discovery only earns its keep for UNKNOWN/NON-generic symmetries
(the toy ladder rungs 12-14). Bridge direction CONCLUDED. Lesson: validate "discovered beats RANDOM (not just
single/base)" on a real downstream task before building production — covering a region (Rung C) != adding value
over random coverage (Rung D). New files: v14_ladder_anchor_grad_test, v14_ladder_move_leakage_test,
feature_gap_proxy_cifar_se2_distribution, aug_training_proxy_cifar, eqm_lite_aug_proxy, _se2_discovery.

## v14 BEAT-CROP POLICY LADDER 2026-06-05 — NOT authorized; policy TIES crop, never beats it.
Goal: learn aug POLICY qθ(T) (anchor+entropy+UTILITY) to BEAT crop, not imitate. Policy = qθ over 2-gen
SE(2) Lie basis M=exp(z1*A1+z2*A2), z~N(0,diag(σ²)). A (policy safety) PASS: anchor grad to full policy
(A1,A2,logsig)=40.6, encoder frozen, no-grad=0, broad hinge no-leak. B (policy support) PASS: ANCHOR_ENTROPY
with a 2D INDUCED-COVARIANCE floor (penalize smaller eigenvalue of sampled (tx,ty) cov) cover 0.25 beats
single 3.66/random 0.62/base 1.43, high-rank 2D (erat 0.29), bounded; ABLATION anchor-only anisotropic
(erat 0.13) -> entropy-on-coeff-std is NOT enough, need the 2D induced-cov floor (generators can be near-
parallel otherwise). C (utility vs crop, classifier broad-±6 robustness, 3 seeds) = NO BEAT: transl±6 base
0.370, KNOWN_CROP 0.377, random_policy 0.371, anchor_only 0.371, anchor_entropy 0.380, anchor+entropy+UTILITY
0.382. Utility policy top-numerically + beats random but TIES crop within noise (0.382 vs 0.377; crop std
0.018 >> 0.005 margin); utility = faint edge over entropy/random, NONE over crop. D (EqM-lite policy)
INCONCLUSIVE: gap 0.0018 within noise (velocity field eps-x locally translation-robust; signal-guard flags it;
not run full). VERDICT: NOT authorized for FID. Mechanism WORKS (A,B) and policy MATCHES crop + beats random
(C) but does NOT beat crop. WHAT BLOCKED: no headroom over near-optimal crop for a KNOWN/GENERIC symmetry.
ARC CLOSED (v12 rotation wrong-family / v13 single under-diversity / v14 distribution==random / v14-policy
ties-crop): for known generic CIFAR translation there is NOTHING to discover beyond crop; discovery's value
is only for UNKNOWN/non-generic symmetries (toy ladder rungs 12-14). EqM bridge direction CONCLUDED. KEY
METHOD LESSON: a learned aug must beat RANDOM *and* the KNOWN hand-coded baseline beyond noise on a real
downstream metric; "covers the region" (Rung B) and "ties crop" (Rung C) are not "beats crop." Commit 3110161.

V15 (model-aware SAFE-ADVERSARIAL aug, 2026-06-05) — STRONGER NEGATIVE than v14: cleverness is WORSE than
crop. Reframe (user): stop imitating crop, BEAT it by targeting current model weakness — anchor=safety,
frozen-scorer-loss=utility/hardness, conditionality=per-image qθ(T|φ(x)) (levers v14 never tested). Rungs:
R1 KNOWN ceiling (3 seeds) crop_pad4=0.409 clear ceiling (tstd .001), beat-crop VALID (quick 1-seed reframe
to transl_scale was noise). R2 frozen-scorer hardness gate PASS (safe-hard CE 1.82>crop 1.70, MORE on-manifold
ED 2.95<3.26, 2D) — *** FALSE POSITIVE for downstream ***. R3 TRAIN w/ global safe-hard (3 seeds) NO BEAT all
lams, MONOTONE harder->worse (l1.5 .397, l3.0 .385<crop .409, l3.0<random .393). R4 CONDITIONAL q(T|φ(x))
(3 seeds) WORSE: .362-.363 << crop .409 (~=base transl), CRUSHES clean acc (.38 vs .47); per-image mu_std
real (3.4-8px x) so adaptivity IS present but HURTS. MECHANISM: adversarial-vs-frozen-scorer finds scorer
BLIND SPOTS not generalizing transforms; anchor keeps images realistic but chosen SHIFTS adversarial-degenerate
-> train/test mismatch. CIFAR translation uniform/isotropic => crop ~Bayes-optimal, no per-image right shift
=> cleverness backfires. CIFAR-translation aug discovery EXHAUSTED v12-v15. Do NOT run FID. METHOD LESSON
(adds to v14): frozen-scorer hardness != training value (a hardness gate can be a false positive); model-aware
adversarial aug can be WORSE than uniform random aug when the nuisance is generic. To make discovery matter on
real data the TESTBED must have a non-generic symmetry crop can't capture (learned-latent rotation / domain
equivariance), not CIFAR translation. Files experiments/v15_rung{1,2,3,4}_*.py + results_v15_rung*.json.
Bridge plan + summary.md + pipeline.json updated. NEXT HUMAN: pick non-generic-symmetry testbed OR close out
bridge+pivot; pending v10_hardneg bridge FID (job 19405301).

V16 (RESIDUAL policy over best-known, VALIDATION-UTILITY objective, 2026-06-06) — 4TH CONSECUTIVE NEGATIVE;
falsifies the LAST untested objective. Reframe (user): drop frozen-scorer hardness (v15 backfired); crop is a
strong BASE; learn a bounded RESIDUAL over BEST_KNOWN optimized for short-run VALIDATION utility (bilevel-lite
ES), anchor(excess-over-base)/entropy as CONSTRAINTS, no per-image conditional. E1 known ceiling (3 seeds,
mixed-corruption robust val=translate+scale+brightness): BEST_KNOWN=crop_pad4_color 0.429 (>crop_pad4 0.420>noise;
color helps b/c val corruption has brightness). E2 residual policy ES-on-val-utility (3 seeds, SEPARATE test
split): NO BEAT vs random -- learned 0.41483 vs rand_residual 0.41458 (delta 0.00025 << noise 0.005); ES grew
ranges r_tx/r_ty 1.0->1.3 but didn't beat theta=0 init; residual CATEGORY beats best_known 0.4008 by +0.014
(~=noise) from added diversity ONLY (free, no learning). E3 stage CURRICULUM ES-on-schedule phi=(a,b),
strength=sigmoid(a+b*frac) scaling fixed residual (3 seeds): NO BEAT -- curriculum 0.4132 <= static 0.4148 =
random_cur 0.4148; ES drifted to mild decay s0.40->0.30, zero benefit. E4 EqM-lite NOT RUN (gated, both no-beat).
VERDICT NOT authorized for FID. KEY: over a known generic base neither residual SHAPE (E2) nor TIMING (E3) can
be learned to beat fixed/random; only gain = 'add a mild residual at all' (~=noise, no learning). The
VALIDATION-UTILITY bilevel target (distinct from v14 anchor-match + v15 hardness) ALSO converges to random.
WHOLE-ARC THESIS DECISIVELY CONFIRMED across objectives {anchor-match, hardness, validation-utility} x forms
{single, distribution, policy, conditional, curriculum}: NO headroom for augmentation discovery on a KNOWN
GENERIC nuisance. To make discovery matter the TESTBED must change (unknown non-generic symmetry crop can't
capture: learned-latent rotation / domain equivariance / dataset w/ real hidden symmetry). METHOD LESSON adds:
a validation-utility (not hardness) bilevel objective still ties random when there's nothing to learn; ES will
report a spurious 'beat' if the margin test isn't noise-aware (fixed: beats_random needs >0.5*noise). Files
experiments/v16_{known_aug_ceiling,residual_policy_proxy,residual_curriculum_proxy}.py + results_v16_exp{1,2,3}.json.
Commits 547776e/8966103 + final. NEXT HUMAN: (a) non-generic-symmetry testbed, (b) close bridge+pivot, or (c)
write up negative as methodological result. Pending v10_hardneg bridge FID (job 19405301).
