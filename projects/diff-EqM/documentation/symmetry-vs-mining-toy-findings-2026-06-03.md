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

## Suggested next step (not yet started)

A toy where the symmetry is *unknown in class* (not just unknown angle) — learn a general
transformation operator `T_ψ` (small net / Lie-group parametrization) instead of a single rotation
angle — and measure whether discovery still ties "known" when the operator family is not handed over.
That is the closest cheap proxy to the IN-1K discovery problem.
