# Finding: symmetry-constraint beats hard-negative mining for off-manifold generalization (toy evidence)

**Date:** 2026-06-03
**Status:** smoke-level viability probe. NOT publishable. Filter for the v10-vs-symmetry direction.
**Scripts:** `projects/diff-EqM/experiments/symmetry_toys/{exp2.py, eqm_toy.py, eqm_toy2.py}` (CPU, seconds–minutes).

---

## Question

The notes propose: an overfit model can still generalize if the *gaps* (off-manifold regions) are
shaped correctly. Two candidate levers for shaping the gaps:

1. **Hard-negative mining** (the current project lever — v10).
2. **Symmetry: discover the rule the data obeys, then enforce it** (constraint and/or augmentation).

Which lever actually installs structure the model never saw? And can the symmetry be *discovered*
unsupervised, or must it be handed over?

Three throwaway toys, each cheap enough to run locally on CPU.

---

## Toy 1 — arithmetic / classification (`exp2.py`)

**Task:** energy `E(a,b,c)`, `a,b∈{0..9}`, `c=a+b`. Hidden rule = commutativity (`a+b=b+a`).
**Split:** train only on `a≤b`; test only on `a>b` (orderings never seen as positives).
**No-leak:** structured negatives carry `c⁻≠c`, so the correct swapped answer is never revealed.
**Overfit premise:** full seen-region negatives → all arms reach train acc 1.000.

| arm | what | train (a≤b) | TEST (a>b) |
|---|---|---|---|
| BASE | overfit positives only | 1.000 | 0.000 |
| STRUCT | + sparse plausible swapped negatives | 1.000 | 0.007 |
| FULLNEG | + full swapped-complement push-up | 1.000 | 0.000 |
| EQUIV | + tie `E(a,b,c)=E(b,a,c)` | 1.000 | **1.000** |

chance = 0.053.

**Verdict:** hard negatives (sparse OR full) do not install the symmetry — `≈0`. The equivariance
*constraint* generalizes perfectly. Same symmetry information, opposite outcome depending on whether
it is used as **negatives** (fails) or a **constraint** (works). FULLNEG also disproves the
"leak-by-elimination" worry — pushing up all wrong answers never pins the correct one *down*.

---

## Toy 2 — EqM-faithful generative, ring manifold (`eqm_toy.py`)

**Mechanism:** EqM skeleton — linear path `x_γ=(1−γ)x+γε`, predict velocity `v=(ε−x)`, sample by
integrating the field from noise. (Same field-matching as the repo; 2D so it runs in minutes.)
**Data:** 8 Gaussian modes on a circle. Hidden rule = 45° rotation.
**Split:** train on 6 modes, hold out 2 (`{2,5}`) that the rotation implies.
**Metric:** `recall@heldout` = fraction of samples landing near the held-out modes (ideal 0.25);
`modes/8` = how many of all 8 modes get ≥1% of samples.

| arm | recall@heldout | modes/8 | note |
|---|---|---|---|
| BASE | 0.000 | 6.0 | gaps empty |
| HARDNEG (v10 analog) | 0.000 | 6.0 | mining fails to fill gaps |
| EQUIV_KNOWN | 0.150 | 8.0 | given the rule → fills gaps |
| EQUIV_DISC | 0.225 | 8.0 | learned 45.5° (true 45°) → fills gaps |

**Note:** EQUIV_DISC appeared to *beat* KNOWN, but this was an uncontrolled confound — DISC's recipe
included a rotated-data augmentation term (weighted 3×) that KNOWN lacked. See Toy 3.
Also: naive discovery first collapsed to identity (359.8°); fixed with an identity barrier +
off-zero init. Discovery is fragile.

---

## Toy 3 — controlled mechanism split (`eqm_toy2.py`)

Separate the two ways a symmetry can be used, **equal weight (W=1.0)** on each, and compare
discovery vs known on an *identical* recipe.

- **CONSTRAINT:** `||f(Rx) − R·f(x)||²` (tie field across the symmetry).
- **AUGMENT:** EqM loss on rotated data (= data-aug into the gaps).

| arm | recall@heldout | modes/8 | note |
|---|---|---|---|
| BASE | 0.000 | 6.0 | |
| HARDNEG | 0.000 | 6.0 | v10 — dead |
| KNOWN_CONSTRAINT | 0.150 | 8.0 | constraint alone works |
| KNOWN_AUG | 0.157 | 8.0 | augment alone works (≈ constraint) |
| KNOWN_BOTH | 0.193 | 8.0 | both > either (mildly additive) |
| DISC_BOTH | 0.194 | 8.0 | learned 45.6°; **ties known** |

**Verdicts:**
1. **Discovery is free.** With an equalized recipe, discovering the symmetry (45.6°) performs
   identically to being handed the true 45°. The earlier "discovery wins" was purely the unequal
   extra term.
2. **Both uses of symmetry work, about equally** (constraint ≈ augment). The lever is the
   *symmetry*, not the specific way it is injected.
3. **Combining helps mildly** (0.193 vs ~0.15).
4. **Hard-negative mining = 0 in every configuration, in both classification and EqM settings.**

---

## Summary verdict

| lever | classification | EqM generative |
|---|---|---|
| hard-negative mining (v10) | dead (≈0) | dead (0.000) |
| symmetry as constraint | perfect (1.000) | works (0.150) |
| symmetry as augmentation | — | works (0.157) |
| **discovering the symmetry** | — | **free — ties known (0.194)** |

The notes' core bet — *overfit, but shape the gaps to generalize beyond the data* — survives a
controlled test. The effective gap-shaper is **symmetry**, not **hard negatives**. And the symmetry
can be **discovered unsupervised at no measured cost** (in this toy).

---

## Implication for the project (v10)

v10 = adaptive hard-negative mining on the EqM regression target. In both toys, hard-negative mining
contributes **nothing** to generalizing onto manifold regions the model never saw. This does **not**
say v10 cannot improve FID — mining can still sharpen an *existing* boundary, which is a different and
weaker claim. It does say: if the goal is *installing manifold structure / generalization*, mining is
the wrong lever, and a symmetry-discovery + equivariance-constraint approach is the higher-leverage
direction worth a proposal.

This is a redirect *signal*, not a gate decision. Do not act on it against the pre-registered phase
plan without a Variant Proposal + PI check.

## Hard caveats (do not over-read)

- All toys are 2D / tiny / single clean symmetry (commutativity, 1-param rotation). Image manifolds
  (IN-1K) have many-parameter, messy, unknown symmetries — discovery there is the real unsolved work
  and is **not** evidenced here.
- EqM arms plateau at recall ≈0.19 vs ideal 0.25 (held-out modes filled but under-weighted, ~77% of
  ideal mass). Coverage is 8/8.
- Discovery was fragile: naive objective collapsed to identity; needed a barrier + off-zero init.
- "Discovery is free" is demonstrated only for a 1-parameter rotation. No claim at scale.
- Smoke-level: ≤3 seeds, no FID, no high-dim confirmation. Per CIFAR/sanity rules, this is a filter,
  not a result.

## Rung 3 — latent symmetry hidden by a nonlinear decoder (`latent_symmetry.py`) — RAN 2026-06-03

**Setup:** latent 8-mode ring (45° symmetry) → frozen random nonlinear decoder `g: R²→R¹⁶` →
observed data where the symmetry is the nonlinear warp `g∘R∘g⁻¹` (not a named linear op). Train EqM
in observed space; hold out modes {2,5}. Operator `T_ψ` learned in observed space (residual MLP);
discovery recipe = closure-aug + bounded-move target + cyclic `T^P≈id` (P=8).
Run on cluster job 18980880 (gpu_requeue, **CUDA**) and locally (CPU) — results match.

| arm | recall@HO | modes/8 | T_on_manifold | role | result |
|---|---|---|---|---|---|
| BASE | 0.000 | 6 | — | negative/floor | empty (expected) |
| ORACLE | 0.164 | 8 | — | **positive control** | fills gaps → harness valid, gap fillable |
| DISC_LINEAR | 0.000 | 6 | 0.00 | **negative control** | linear op can't stay on nonlinear manifold |
| DISC_NONLIN | 0.000 | 6 | 0.14 | **treatment** | **FAILS** — recall 0, mostly off-manifold, σ scattered |

**Verdict: rung 3 FAILED.** Treatment ≈ negative control. ORACLE confirms it is not a harness/metric
problem — the gap is fillable. A **free observed-space operator `T_ψ` cannot discover the
nonlinearly-hidden symmetry**: nothing pins `T(x)` onto the *data* manifold (closure-aug is too weak —
the field just fits whatever T emits), so T drifts off-manifold; cyclic `T^P=id` is satisfiable by
near-identity scattered maps.

**Lesson (rhymes with the arithmetic rung):** the structural prior dismissed as overengineering —
**enc → linear op → dec** (autoencode to a latent where the symmetry is linear) — is likely the
load-bearing piece for nonlinearly-hidden symmetries. Free-form discovery in observed space is not
enough.

**Process note:** the 2-term recipe (aug + 1/d² barrier) failed local smoke first (T flew
off-manifold); cyclic + bounded-move were added as evidence-based fixes — and STILL failed. Two
strikes on observed-space free-operator discovery. Per kill rule, do not retune this form again;
change the mechanism (enc-linear-dec) for the next rung.

## Rung 4 — latent-coordinate symmetry `T=dec(M·enc(x))` (`latent_symmetry_rung4.py`) — RAN 2026-06-03

**Thesis under test:** the symmetry is not discoverable as a free observed-space operator (rung 3),
but may become a simple LINEAR operator in *learned latent coordinates*. Same world/data/split/
metrics/seeds as rung 3; only new arm DISC_LATENT = `dec(M·enc(x))`, `M` one global learned 2×2.
Discovery loss recipe held FIXED vs rung-3 DISC_NONLIN (closure-aug + bounded-move + cyclic `T^P≈I`)
so any difference is due to PARAMETRIZATION. AE-first protocol; `M` = one global matrix (anti-
memorization); held-out never in training.

**Process note (a real methodology bug, caught + fixed):** first run used *semi-freeze* (recon
anchor during the joint phase). The symmetry terms overpowered the anchor and CORRUPTED the AE:
recon rel 0.03 (alone) → 0.44 (after joint). That run was INCONCLUSIVE (failed its own precondition:
good recon during symmetry search), cluster job 18986548 marked invalid. Fix = FULLY freeze enc/dec
after a verified-good pretrain (recon-only converges to ~0.030 rel at K=2), then search `M` + field in
the fixed latent. (Lesson logged: verify the AE reconstructs BEFORE judging the symmetry; a corrupted
AE masquerades as a thesis negative.)

**Corrected run (frozen AE, local CPU; cluster cross-device confirm pending 2FA re-auth):**

| arm | recall@HO | modes/8 | T_on_man | AE_rel | ‖M−I‖ | ‖M⁸−I‖ | role/result |
|---|---|---|---|---|---|---|---|
| BASE | 0.000 | 6 | — | — | — | — | floor |
| ORACLE | 0.179 | 8 | — | — | — | — | **positive control ✓** |
| DISC_LINEAR | 0.000 | 6 | 0.00 | — | — | — | negative control ✓ |
| DISC_NONLIN | 0.000 | 6 | 0.08 | — | — | — | rung-3 free op, fails ✓ |
| DISC_LATENT | 0.000 | 6 | 1.00 | **0.034** | **0.01** | 0.074 | **FAILS — M→identity** |

**Verdict: rung 4 FAILS — and it is a GENUINE negative (recon good, not an AE failure).** DISC_LATENT
recon is verified-good (0.034) yet `M` collapses to identity (‖M−I‖=0.01) → T≈recon≈x → on-manifold
but inert → recall 0.

**Root cause, rigorously confirmed (latent-geometry diagnostic):** encode the 8 mode centers into the
frozen recon latent — it is a **distorted loop, NOT a circle**: latent radii 2.57–9.40 (a circle would
be constant), angular gaps {49,69,25,49,57,39,51,21}° (not 45°). The 45° rotation is therefore
**nonlinear in the reconstruction-faithful latent**; no single linear `M` preserves that distorted
loop, so `M→I` is the correct optimum. **Reconstruction alone does not shape latent geometry to
linearize the symmetry** — it yields an arbitrary nonlinear reparametrization of the manifold.

**Kill rule fires:** "DISC_LATENT = BASE while AE recon good → stop this mechanism, rethink." Stop the
recon-frozen-latent + linear-M mechanism.

## Ladder status after rung 4

| rung | mechanism | result |
|---|---|---|
| 1 arithmetic | equivariance constraint vs hard-neg | constraint perfect; hard-neg dead |
| 2 ring (linear-in-obs symmetry) | known + discovered rotation | works; discovery ties known |
| 3 nonlinear-hidden, free obs-space op | residual MLP `T` | FAILS (off-manifold) |
| 4 nonlinear-hidden, enc-linear-dec, **recon-frozen** latent | `dec(M·enc)` | FAILS (recon latent ≠ linearizing latent; M→I) |

Consistent theme: **discovery works iff the operator family is expressible in the coordinates you
search.** Rung 2 worked because the ring symmetry IS linear in observed 2D. Rungs 3–4 fail because the
symmetry is nonlinear in both observed space and the recon latent, and neither a free observed MLP nor
a linear-map-in-recon-latent reaches a coordinate system where it is simple.

## Rung 5 — shape the latent + the decisive ORACLE_LATENT control (`latent_symmetry_rung5.py`) — RAN 2026-06-04

Pursued 5A (shape latent by symmetry: DISC_JOINT, modes scratch/warm/alt) BUT added the single most
informative control: **ORACLE_LATENT** = encoder SUPERVISED to the true latent z (clean circle),
frozen, dec = true world decoder, then run the exact discovery recipe to learn M. Asks: *given a
perfect clean latent, does the recipe even discover the 45° rotation?*

Result (all 3 joint-modes, consistent):

| arm | recall@HO | enc→z err | ‖M−I‖ | M_angle | result |
|---|---|---|---|---|---|
| ORACLE (pos ctrl) | 0.178 | — | — | — | fills ✓ |
| **ORACLE_LATENT** | **0.000** | **0.011** (perfect latent) | 0.09 | **≈ −0.5°** | **M → identity** |
| DISC_JOINT (scratch/warm/alt) | 0.000 | — | scattered | scattered | fails (recon 0.4–0.7) |

**This reframes rungs 3–5.** Even with a PERFECT clean-circle latent (enc→z 0.011), the recipe leaves
M at the identity (angle ≈0, recall 0). So the blocker is **NOT (only) latent geometry — the discovery
RECIPE is broken.**

**Root cause:** the anti-identity term was `(move − target)²`. Its gradient **vanishes at the identity**
(squared → flat at move=0), so M cannot escape identity; and the closure term actively pulls M back
toward identity (moving lands on uncovered held-out → high closure). Chicken-egg lock in the
*optimizer*: M won't move until the field covers held-out; field won't cover until M moves. Rung 2
(which worked) effectively avoided this via an explicit identity-barrier with nonzero gradient at 0; I
regressed when I switched to bounded-move in rungs 4–5.

**Methodological win:** the ORACLE_LATENT control is what isolated this — without it, rung-4's failure
looked like a latent-geometry/thesis negative; with it, the true culprit (recipe) is exposed. Positive
controls earn their cost (the CLAUDE.md rule, again).

## Rung 6 — recipe fix: hinge anti-identity (`latent_symmetry_rung6.py`) — RUNNING 2026-06-04

One change: replace `(move−target)²` with `relu(target−move)` (linear hinge → constant nonzero push
off identity until move≥target), bolder M init. Everything else identical. Quick smoke already shows
ORACLE_LATENT M escaping identity (angle −0.4°→−5.5° at 300 steps, σ entering a held-out mode).
**Gate:** ORACLE_LATENT must recover M_angle ≈ ±45° (or multiple) and fill held-out → recipe validated;
THEN DISC_JOINT (unsupervised) judged. Full 3-mode parallel run in progress.

**Rung 6 RESULT (all 3 modes):** hinge INSUFFICIENT. ORACLE_LATENT M_angle 5.2°/4.7°/3.0° — still
≈identity, recall 0, even with the perfect latent (enc→z 0.011). M jitters near identity (quick run
−5.5° → full +3°), not escaping.

**Force-balance diagnostic** (clean latent, sweep the anti-identity weight W_MOVE ∈ {0.5,2,5,10,20}):
M_angle tops out at −12° to −24°, **never reaches ±45°**, recall ~0 at every weight. So it is **NOT a
force-balance / weak-knob problem.**

**Root cause (real finding): discrete modes are ill-posed for gradient-based symmetry discovery.**
Stronger push moves M off identity but it STALLS partway (~−15°) and cannot reach 45°, because with
*discrete* modes the intermediate rotations (15–44°) map samples BETWEEN modes = off-manifold = high
closure penalty. There is an **energy barrier between identity and the true symmetry** — no continuous
on-manifold path for gradient descent to walk M from 0°→45°. M is trapped in the identity basin.
This (not latent geometry, not gradient-kink) is why rungs 3–6 fail.

## Rung 7 (next) — continuous manifold so continuity forces the symmetry

Replace the 8 discrete modes with a CONTINUOUS ring (dense angle φ ∈ [0,2π) with a held-out ARC, e.g.
φ ∉ [φ_a, φ_b]), same nonlinear decoder, EqM in observed space. Now rotating φ stays ON the manifold
the whole way (a circle is rotation-closed), so there is a smooth on-manifold path from identity to the
symmetry → no barrier → M can continuously rotate to discover it; continuity across the gap forces the
symmetry to extend into the held-out arc. Same ORACLE_LATENT control + DISC arms + hinge recipe.
recall@heldout = fraction of generated samples whose latent angle lands in the held-out arc.
Gate: ORACLE_LATENT must now recover a rotation + fill the arc (proving discreteness was the blocker);
then DISC_JOINT is the real unsupervised test.

**Rung 7 RESULT (continuous, scratch full):** continuity did NOT fix it.

| arm | recall_arc | bins/36 | onman_gen | ‖M−I‖ | M_angle | note |
|---|---|---|---|---|---|---|
| BASE | 0.002 | 32 | 0.98 | — | — | floor; field models continuous ring well |
| ORACLE (random SO(2) aug) | 0.063 | 36 | 0.98 | — | — | positive control: arc fillable ✓ |
| DISC_LINEAR | 0.002 | 32 | 0.48 | — | — | neg control ✓ |
| **ORACLE_LATENT** | **0.001** | 32 | 0.98 | **0.01** | **−0.0°** | **M → identity AGAIN** (clean latent, enc→z 0.022) |
| DISC_JOINT | 0.001 | 32 | 0.57 | 1.26 | 64° | recon 0.43, fails |

Continuity removed the closure barrier (all rotations are on-manifold), but a NEW identity-attractor
took over: the cyclic `M^P≈I` term pulls M to the **nearest finite-order matrix = identity**. The
attractor moved (discrete→closure-barrier; continuous→cyclic-pull) but the destination is the same.

## SYNTHESIS — identity is a universal attractor for unsupervised symmetry discovery

Across rungs 5–7, on discrete AND continuous manifolds, with clean (ORACLE_LATENT) AND learned latents,
under multiple anti-identity mechanisms (squared-move, hinge-move, 20× push, cyclic), the learned
operator **collapses to the identity**. Reason: the identity is a *valid symmetry* — it satisfies every
structural constraint we can write without naming the group (on-manifold, finite-order, reconstruction).
The only term opposing it ("must move") is a soft penalty that either has vanishing gradient at I, gets
trapped behind an off-manifold barrier, or is overpowered by a finite-order term that itself prefers I.

Positive controls bound this cleanly: ORACLE (given the true group) fills the gap; ORACLE_LATENT (clean
latent, but discover the operator) does NOT. So the blocker is the **discovery objective**, not the
latent, the manifold continuity, or the harness.

**Conclusion (answers the original question + the prior-budget question):** you cannot discover a
nontrivial symmetry "for free." Some structural prior that *excludes the identity* (normalize the
generator, or restrict to a nontrivial group class) appears necessary — soft penalties don't suffice.
This is the empirical answer to "how much prior is cheating": *enough to exclude the identity is not
optional.* Rung 8 tests the minimal such prior (constrain ‖M−I‖ to a fixed nonzero size, direction
free — excludes identity WITHOUT asserting "it's a rotation").

## Rung 8 — minimal identity-exclusion (`latent_symmetry_rung8.py`) — RAN 2026-06-04

Hard-project M onto ‖M−I‖=1.4 each step (excludes identity, direction free); drop soft move+cyclic;
closure-aug only.

| arm | recall_arc | onman_gen | ‖M−I‖ | ‖M⁸−I‖ | M_angle | result |
|---|---|---|---|---|---|---|
| ORACLE (true group) | 0.066 | 0.98 | — | — | — | fills arc ✓ |
| **ORACLE_LATENT** | **0.001** | 0.74 | 1.40 (held) | **321** | 25.6° | **FAILS — M is a non-rotation** |
| DISC_JOINT | 0.001 | 0.48 | 1.40 | 96 | 61° | fails |

Identity-exclusion did NOT rescue discovery. With identity forbidden, M drifts to an **arbitrary
fixed-norm matrix** (‖M⁸−I‖=321 → expanding/shearing, NOT the manifold-preserving rotation) and the
arc stays empty.

**DEEPEST ROOT CAUSE (the ladder's terminal finding): closure-via-field cannot anchor the operator to
the data manifold, because the field co-adapts.** `closure = eqm(f, T(x))` is minimized by training the
field `f` to model wherever `T(x)` lands — NOT by forcing `T(x)` onto the true data manifold. So the
closure objective is satisfiable by *any* operator; nothing holds T to the manifold. ORACLE works only
because it injects the TRUE group elements (`DECODE(R_true·z)`), giving the field real on-manifold
targets. Discovery has no such anchor: operator + field co-adapt to trivially satisfy closure.

## LADDER COMPLETE — final verdict (rungs 1–8)

| rung | question | result |
|---|---|---|
| 1 | equivariance vs hard-neg (arithmetic) | constraint perfect; **mining dead** |
| 2 | known/discovered rotation (symmetry linear in observed) | both work; discovery ties known |
| 3 | free observed-space operator, nonlinear-hidden symmetry | fails (off-manifold) |
| 4 | enc-linear-dec, recon-frozen latent | fails (recon latent ≠ linearizing latent) |
| 5 | ORACLE_LATENT control | recipe bug exposed: M→identity even with perfect latent |
| 6 | hinge anti-identity | insufficient (M jitters near identity) |
| 7 | continuous manifold | identity-attractor persists (cyclic pulls to I) |
| 8 | minimal identity-exclusion | M drifts to non-symmetry; **closure can't anchor to manifold** |

**Bottom line.** Hard-negative mining (v10's mechanism) installs no manifold structure in any setting.
Symmetry **constraints** generalize **when the symmetry/group is known** (ORACLE, rung-1 equivariance,
rung-2 known rotation). **Unsupervised discovery** of a nonlinearly-hidden symmetry, via a learned
operator + closure/aug objective, **does not work** — two compounding obstacles: (a) identity is a
universal attractor that prior-free constraints can't break, and (b) even with identity excluded, a
co-adapting field cannot anchor the operator to the data manifold, so the operator drifts to a
non-symmetry. A working formulation would need a **fixed manifold reference** (distribution-matching to
frozen data, not field-closure) and/or a **known group class** — i.e., it cannot be fully unsupervised
as posed.

## Implication for diff-EqM / v10 (project decision — needs PI/user)

- v10 = adaptive hard-negative mining. Across every toy, mining installs zero new manifold structure;
  it can only sharpen an EXISTING boundary. This is consistent with v10's small measured effects.
- The higher-leverage alternative (symmetry/equivariance) only pays off when the symmetry is **known
  or cheaply specified** — NOT discovered. For IN-1K images, useful symmetries (crops, flips, color)
  are known and could be injected as equivariance constraints/augmentation directly, skipping discovery.
- Recommended framing for a paper: "symmetry constraints > hard-negative mining for generalization in
  regression-target generative models, **when the symmetry is specified**" — and report unsupervised
  discovery as an analyzed negative (identity-collapse + no manifold anchor). This is honest and still
  publishable; do NOT claim unsupervised latent-symmetry discovery.

## Rung 9 — FROZEN-ANCHOR discovery (`latent_symmetry_rung9.py`) — RAN 2026-06-04

Directly attacks the rung-8 terminal diagnosis (field co-adaptation): learn T against a FROZEN,
non-co-adapting manifold reference (energy-distance to UNLABELED full-manifold samples) BEFORE EqM,
freeze T, then use it as EqM augmentation. Three data roles: anchor (full manifold, unlabeled, no EqM
targets), eqm-train (visible only), eval (full). Arms add FIELD_CLOSURE_DISC (reproduce old failure)
and VISIBLE_ANCHOR_DISC (anchor on visible only — leakage/negative control).

| arm | recall_arc | T_onman | shift_std | mmd_full | read |
|---|---|---|---|---|---|
| ORACLE | 0.068 | — | — | 0.025 | positive control ✓ |
| FIELD_CLOSURE_DISC | 0.002 | **0.00** | 99° | 0.73 | reproduces failure: T off-manifold ✓ |
| **FROZEN_ANCHOR_DISC** | **0.009** | **0.70** | **95°** | 0.065 | anchor fixes on-manifold; arc NOT filled |
| VISIBLE_ANCHOR_DISC | 0.002 | 0.64 | 97° | 0.066 | ≈ frozen → no full-manifold advantage |

**Result: PARTIAL — the hypothesis is half-right.**
- **CONFIRMED:** the frozen anchor fixes co-adaptation. FIELD_CLOSURE leaves T off-manifold
  (T_onman=0.00); FROZEN_ANCHOR holds T ON the manifold (T_onman=0.70). The rung-8 diagnosis was
  correct and the frozen anchor is the right fix *for that problem*.
- **NEW FAILURE:** distributional matching (energy-distance/MMD) **under-constrains the operator**. T
  becomes a distribution-preserving RANDOM SHUFFLE on the manifold (shift_std≈95°, near-uniform), NOT
  a coherent rotation/flow. So it transfers no usable structure → arc not filled (0.009 vs ORACLE
  0.068). FROZEN≈VISIBLE confirms no symmetry was actually learned (no benefit from full-manifold
  support — T isn't using manifold structure, just shuffling).

**Refined conclusion.** A frozen manifold anchor is NECESSARY (fixes co-adaptation) but NOT SUFFICIENT.
Missing ingredient = operator COHERENCE: T must be a consistent flow (same directional shift for all
points), not merely distribution-preserving. Candidate rung 10 (NOT run; one mechanism): frozen anchor
+ coherence — parametrize T as a single Lie-generator flow `exp(tξ)`, or penalize shift-inconsistency
(var of induced Δφ), so the only distribution-preserving non-identity operators are coherent flows.

## Ladder status (rungs 1–9)

mining dead everywhere · symmetry constraints work when KNOWN · unsupervised discovery blocked by a
3-layer stack, now mostly diagnosed: (1) identity attractor, (2) field co-adaptation [FIXED by frozen
anchor, rung 9], (3) operator-coherence under distributional anchoring [OPEN]. Frozen anchor is real
progress; coherence is the remaining gap.

## Rung 10 — frozen anchor + COHERENT operator (`latent_symmetry_rung10.py`) — RAN 2026-06-04

One mechanism change from rung 9: make the operator coherent BY CONSTRUCTION — a single latent matrix
`T(x)=dec(M·enc(x))`, `M` 2×2 applied to every point (one consistent shift) — instead of a free
residual MLP. Trained against the same frozen energy-distance anchor, frozen, then EqM augmentation.

| arm | recall_arc | T_onman | shift_std | M_angle | read |
|---|---|---|---|---|---|
| ORACLE (full group) | 0.068 | — | — | — | positive control |
| BASE | 0.002 | — | — | — | floor |
| FROZEN_RESIDUAL (rung 9) | 0.009 | 0.70 | 95° | — | incoherent shuffle |
| **FROZEN_LATENT_CLEAN** | **0.025±0.034** | 0.71 | **25°** | **−38.8°** | coherent; recovers ~rotation |
| FROZEN_LATENT_DISC | 0.013 | 0.42 | 25° | −24.3° | coherent, weaker (learned dec) |

**Result: PARTIAL SUCCESS — best discovery result of the ladder.** The single-matrix operator fixed
coherence (shift_std 95°→25°) and **recovered a real rotation (M_angle ≈ −39°)**. Recall climbed
floor(0.002) → residual(0.009) → coherent(0.025) — 12× floor — so with all three blockers addressed
(identity-exclusion via move + frozen anchor for co-adaptation + single-matrix for coherence),
discovery DOES transfer real supervision into the held-out arc.

**Not full ORACLE parity** (0.025 vs 0.068, variance ±0.034): (a) a SINGLE learned rotation covers only
the slice of the arc it maps to, whereas ORACLE used the full random-rotation group; (b) the latent is
only approximately clean (noisy ring → M-rotation maps with spread, shift_std 25 not ~0); (c)
seed-variable. The DISC (fully unsupervised, learned dec) variant is coherent but weaker than CLEAN.

**Net:** the 3-layer blocker is resolved in principle — coherence was the last missing piece.
Unsupervised-ish discovery works partially. Closing the gap to ORACLE is now a COVERAGE problem (one
generator vs the full group), not a discovery problem.

## Ladder status (rungs 1–10)
mining dead · constraints work when KNOWN · unsupervised discovery: 3 blockers (identity / co-adaptation
/ coherence) each diagnosed and individually fixed (rung 8 → move/identity-exclusion, rung 9 → frozen
anchor, rung 10 → single-matrix coherence); result = PARTIAL discovery (recall 12× floor, real rotation
recovered, ~37% of ORACLE). Remaining gap = single-element vs full-group coverage + imperfect latent.
Candidate rung 11: augment with the OPERATOR ORBIT {M, M², …, Mᴾ} (approximate the group from the one
discovered generator) to cover the full arc → should approach ORACLE.

## Rung 11 — orbit augmentation (`latent_symmetry_rung11.py`) — RAN 2026-06-04

Augment EqM with the orbit {M¹..Mᴾ⁻¹} of the single discovered generator (approximate the group).

| arm | recall_arc | bins/36 | M_angle | read |
|---|---|---|---|---|
| ORACLE | 0.068 | 36 | — | positive control |
| FROZEN_LATENT_SINGLE | 0.024±0.033 | 33.3 | −38.6° | rung-10 single aug |
| FROZEN_LATENT_ORBIT | 0.024±0.032 | 33.0 | −38.6° | **identical to SINGLE — orbit adds nothing** |
| FROZEN_LATENT_DISC_ORBIT | 0.019±0.013 | 29.3 | −24.1° | unsupervised, weaker |

**Result: coverage hypothesis REJECTED.** Orbit aug ≡ single aug (0.024). The gap to ORACLE is NOT
coverage — both cover ~33 bins; they differ only in arc *mass*. The residual gap is **operator
PRECISION**: the discovered M is an approximate, seed-variable rotation (shift_std 25°≠0, M_off 1.04,
variance ±0.033); Mᵏ compounds the imperfection so the orbit doesn't tile cleanly, and T transfers
supervision into the gap *fuzzily* where ORACLE's exact rotations deliver clean mass.

## FINAL VERDICT — ladder rungs 1–11

The discovery question is answered. **Unsupervised symmetry discovery for EqM is possible IN PRINCIPLE
but IMPRECISE in practice.** Four blockers, each diagnosed by a control and individually fixed:
1. identity attractor → identity-exclusion / move term (rung 8)
2. field co-adaptation → frozen data anchor (rung 9) [the key insight]
3. operator incoherence → single latent matrix (rung 10)
4. operator imprecision → OPEN (rung 11: not coverage; the discovered rotation is fuzzy/seed-variable)

Result: discovery recovers a real rotation and fills the held-out arc to ~35% of ORACLE (12× floor),
but does NOT reach known-symmetry quality. The remaining lever is operator PRECISION (tighter latent,
orthogonality regularization on M, a sharper anchor than energy-distance) — incremental tuning, not a
new mechanism; deliberately NOT chased here (diminishing returns, rabbit-hole risk).

**Bottom line (stable across 11 rungs):**
- Hard-negative mining (v10 mechanism): installs NO manifold structure. Dead.
- KNOWN-symmetry constraint/augmentation: clean generalization (ORACLE, equivariance). Best lever.
- UNSUPERVISED discovery: works partially (frozen anchor + coherent single matrix), but imprecise →
  below known-symmetry quality. Publishable as an analyzed positive-partial + negative, NOT as a
  working unsupervised method.

**For diff-EqM / v10:** prefer injecting KNOWN image symmetries (crops/flips/color) as equivariance/
augmentation over more hard-negative mining. Do not build unsupervised symmetry discovery into the
paper's critical path.

## INTERPRETATION REVISION (2026-06-04) — the prior "deprioritize / mostly failed" read was too harsh

The ladder does NOT rule out near-oracle latent symmetry discovery. It rules out the NAIVE forms:
discovery through live EqM field-closure (co-adaptation) and free-form/incoherent operators. Once
discovery is (a) anchored to a FROZEN manifold reference and (b) constrained to a COHERENT operator
family, the model BEGINS TO RECOVER A REAL SYMMETRY (rung 10: M≈−39°, recall 12× floor, ~35% ORACLE).
That is a positive signal, not a dead mechanism.

The remaining gap to oracle is **operator PRECISION and representation geometry, NOT theoretical
impossibility.** Rung 11 ruled out coverage (orbit aug ≡ single). The open hypothesis: the discovered
operator is too imprecise/fuzzy (angle ≈ right but not exact; powers compound the error). The next test
is a more precise, group-structured operator parameterization — NOT more priors handing over the answer.

Revised stance: unsupervised latent-symmetry discovery is an ACTIVE, promising direction; pursue
operator precision before judging it. (rung 12 below.)

## Rung 12 — group-structured generator (`latent_symmetry_rung12_group_generator.py`) — IN PROGRESS

Replace the free latent matrix M with `M = matrix_exp(θ·A)` (a smooth group-like action; generator A and
step θ learned, direction NOT specified). Tests whether group structure tightens the operator from
≈−39° toward oracle-like −45° and raises recall toward ORACLE. Same frozen anchor; same controls.
Arms: BASE / ORACLE / FROZEN_LATENT_MATRIX (rung-10 baseline) / FROZEN_LATENT_GEN_SKEW (A skew →
rotation family, strong prior) / FROZEN_LATENT_GEN_FREE (A general + det≈1/orthogonality reg, weak
prior — tests whether precision needs the strong prior or emerges).
