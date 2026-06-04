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
