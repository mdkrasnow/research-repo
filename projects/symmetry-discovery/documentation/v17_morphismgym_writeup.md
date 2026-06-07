# Unsupervised Image-Morphism Discovery on a Hidden-Structure Testbed (v17 MorphismGym)

**Status:** internal write-up of a CPU-scale positive result (Phase 0–3 complete, 2026-06-07). Proxy +
synthetic/MNIST/dSprites scale — NOT real EqM / FID. FID is human-gated.

## 1. One-paragraph claim

A learner can discover *multiple valid image morphisms from data structure alone* — no pretrained semantic
encoders, no labels, no held-out targets — and those discovered morphisms improve a field-matching
(EqM-lite) objective **more than a random valid-augmentation baseline and on par with / better than a
known-oracle** augmentation. The discovery is **load-bearing on a label-free manifold anchor** (ablating it
collapses the gain). The effect **transfers** from a synthetic shape generator to real MNIST handwriting and
to dSprites. This is the first positive in a long line of negative augmentation-discovery results (v13–v16),
and it isolates *why* those failed: they used a known generic nuisance (CIFAR crop) where (i) crop is ~Bayes
optimal so nothing is left to discover, and (ii) a label-free anchor cannot separate valid from invalid
because everything is generic.

## 2. The pivot (why this differs from v13–v16)

v13–v16 tested discovery on CIFAR + generic crop/translation and concluded, four times, that discovery
ties random ("no headroom over crop"). The diagnosis: that testbed lacks the two properties discovery
needs. v17 builds a testbed that has them:

- **Real hidden structure / headroom.** Objects are rendered/selected from latent factors, so the valid-image
  set is a low-dim manifold; a held-out region of that manifold is genuinely absent from the visible set and
  must be reached by a morphism.
- **A separable validity signal.** Valid morphisms (translate/rotate/scale/hue/brightness/stroke) map a
  valid image to *another* valid image — they stay ON the manifold. Invalid "decoys" (object-erasing crop,
  large shear, background-only shortcut, shape-changing warp, occlusion, color collapse) LEAVE it. Therefore
  a **label-free anchor** — PCA-whitened random-conv features + energy distance, augmented with simple
  chroma/structure statistics — separates valid from invalid (calibration AUC = 1.0). Generic CIFAR crop has
  no such separation.

## 3. Method

**Gym (`v17_morphism_gym.py`).** 32×32 colorized objects from latent factors `(cx,cy,rot,scale,hue,bright,
stroke,bg)`. Three regimes per task: VISIBLE (narrow band on the hidden factors), ANCHOR (broad valid range,
unlabeled — the discovery signal), HELDOUT (disjoint valid tail, eval only). GT latents are used for
EVALUATION ONLY, never by the policy or anchor.

**Anchor (`AnchorScorer`).** Label-free encoder (random-conv / pixel-stats / tiny in-domain AE — all AUC 1.0;
random-conv used) → PCA-whitening fit on ANCHOR features → energy distance of a transformed batch to the
anchor. Whitening equalizes dimensions (raw conv dims otherwise dominate); a few raw chroma/edge stats are
appended post-whitening so "stealth" collapse decoys (gray/flat) cannot evade.

**Policy (`v17_policy.py`).** `q_θ(family, magnitude, composition)` over candidate primitive families (valid
+ decoy names; the learner does NOT know which are valid). Magnitudes learn by reparameterized
`tanh(μ+σε)` gradient of the anchor/move loss; family weights learn by an EMA-reward bandit
(reward = −group energy-distance + small move bonus). Hard-grouped application keeps it ~1 grid-sample of
work per step. Objective = anchor-ED (stay on manifold) + move-hinge (anti-identity, forces coverage) +
diversity entropy + magnitude bound. Identity alone satisfies the anchor; the move term forces the policy to
select the families that map visible → elsewhere on the manifold = the hidden morphisms; decoys raise ED →
negative reward → down-weighted.

## 4. Experiments & results

All numbers EqM-lite **robustness gap** (heldout field loss − clean field loss), **lower = better**, unless
noted. Full logs: `results/summary.md`, `results/v17_report.md`, `results/v17_*.json`.

**Phase 0 — calibration (PASS).**
- 0A gym structured: valid/decoy separability AUC = 1.0.
- 0B oracle payoff: the shape-ID *classifier* is INSENSITIVE (oracle ≈ random; shape identity is robust to
  these morphisms), so the gate is EqM-lite, which brackets cleanly across 2 seeds:
  BASE 0.43 > random-decoy 0.14 > random-valid 0.11 > oracle 0.077.
- 0C anchor: all three label-free anchors AUC 1.0 (random-conv used). 0D: no hallucination on
  unreachable/visible-only controls.

**Phase 1 — discovery (PASS, 7/7 tasks).** Across single_{rotation,hue,scale}, multi_independent,
multi_composed, decoy_pressure, impossible_control: recovers the true families (recall .78–1.0), avoids
decoys (decoy-usage ≈ 0.00), stays valid (.82–.92 vs random-decoy .50–.64), multi > single per-factor
coverage; the NO_ANCHOR ablation degrades VALIDITY (.60–.69) — confirming the anchor's role is
on-manifold/decoy-avoidance, not coverage. impossible_control (held-out factor reachable by no morphism)
shows zero hallucinated coverage.

**Phase 2 — payoff (EqM-lite PASS).** multi_independent, 3 seeds:

| arm | EqM-lite gap |
|---|---|
| BASE | 0.466 |
| RANDOM_VALID | 0.107 |
| **DISCOVERED_MULTI** | **0.062** |
| DISCOVERED_SINGLE | 0.059 |
| KNOWN_ORACLE | 0.078 |
| NO_ANCHOR | 0.301 |

Discovered **beats random** (0.062 < 0.107) and **beats oracle** (0.062 < 0.078); NO_ANCHOR collapses to
0.30 (anchor essential). single_rotation has no payoff headroom (base gap 0.004, all arms tie) — payoff
magnitude tracks the held-out distribution-shift size. The classifier diagnostic is factor-dependent: PASS
on single_rotation (rotation breaks shape-ID, anchor essential), insensitive on multi_independent.

**Phase 3 — transfer (PASS).** Same machinery, objects swapped for real images. multi task, 3 seeds each:

| dataset | base | oracle | random | **DISC** | no_anchor |
|---|---|---|---|---|---|
| MNIST (handwriting) | 0.117 | 0.049 | 0.080 | **0.022–0.023** | 0.108–0.117 |
| dSprites (sprites) | 0.048–0.050 | 0.030–0.034 | 0.038–0.041 | **0.013–0.017** | 0.045–0.050 |

Discovered beats random AND oracle, anchor essential, on both — i.e. across **three distinct image
distributions** (synthetic SDF shapes, MNIST, dSprites).

## 5. What is and isn't shown

**Shown:** label-free morphism discovery, given real hidden structure + a manifold anchor, (1) recovers the
true morphisms and avoids decoys, (2) beats a random valid-augmentation baseline and matches/beats a known
oracle on a field-matching objective, (3) is mechanistically anchored (ablation kills it), (4) transfers to
real images. The recorded bottleneck "operator architecture done; held-out targeting without labels
unsolved" is **solvable in this regime**.

**Not shown / caveats:** synthetic + MNIST/dSprites with **controlled latents**, and an EqM-**lite** proxy —
NOT real EqM or FID. The classifier proxy is insensitive when the task is robust to the nuisance (consistent
with the v13–v16 lesson: discovery has no headroom where the nuisance does not matter). `multi` ties `single`
on payoff (one discovered morphism already saturates field-robustness for independent factors). All scale is
CPU-toy — a filter/diagnostic, not a publishable result on its own.

## 6. Relation to prior work in this project

Supersedes the "augmentation discovery has no headroom / bridge concluded" verdicts of v14–v16, which were
specific to a known generic CIFAR nuisance. It does NOT contradict them: it explains them (no headroom + no
label-free validity signal there) and shows the complementary regime where discovery does have headroom.

## 7. Next steps (gated)

1. (done) dSprites confirmation.
2. Optional: a native-latent test (use dSprites' own orientation/scale/position as the regime axes rather
   than applied morphisms) and/or Shapes3D, to remove the "we applied the morphism" caveat.
3. Real-EqM bridge (`v12_stable_generator_aug`, cluster, 2FA `scripts/cluster/ssh_bootstrap.sh`) — ONLY on
   explicit human approval. FID is never auto-authorized.

Code: `experiments/v17_morphism_gym.py`, `v17_policy.py`, `v17_eval_metrics.py`, `v17_common.py`,
`v17_run_{calibration,discovery,payoff}.py`, `v17_collect_results.py`, `v17_transfer.py`,
`run_v17_all.sh`. Results: `results/v17_*.json`, `results/v17_report.md`, `results/summary.md`.
