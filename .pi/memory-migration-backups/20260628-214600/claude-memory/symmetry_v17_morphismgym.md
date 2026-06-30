---
name: symmetry_v17_morphismgym
description: v17 MorphismGym — first POSITIVE in the v13-v17 arc; unsupervised morphism discovery beats random + approaches oracle on EqM-lite in a hidden-structure gym
metadata: 
  node_type: memory
  type: project
  originSessionId: 810dd159-6163-4437-bd01-00ce413d840a
---

v17 MorphismGym (2026-06-06, `projects/symmetry-discovery/experiments/v17_*.py`) is the PIVOT that broke
the v13-16 negative streak (see [[symmetry_v16_distrust]] — those negatives were distrusted/invalid).

**The pivot:** v13-16 tested discovery on CIFAR + a KNOWN GENERIC nuisance (crop/translation) where crop is
~Bayes-optimal -> nothing to discover AND a label-free anchor cannot separate valid/invalid. v17 changed
the TESTBED: a controlled rendered-shape gym (6 shapes x latent factors cx,cy,rot,scale,hue,bright,stroke,
bg) where valid morphisms are REAL but HIDDEN (visible narrow / anchor broad-unlabeled / heldout disjoint
tail eval-only). KEY property: valid morphisms keep images ON the rendered-shape manifold; decoys
(crop-erase/shear/bg-shortcut/shape-warp/occlude/color-collapse) LEAVE it -> a LABEL-FREE PCA-whitened
random-conv energy-distance anchor (+ chroma/structure stats) separates valid/invalid at AUC 1.0. NO
pretrained semantics; GT latents eval-only. Policy qtheta(family,magnitude,composition): magnitudes by
reparam, family weights by an EMA-reward bandit on grouped application (fast; the differentiable-over-all-K
Gumbel stack was ~3x slower in backward and was replaced).

**Results (Phase 0-2 PASS):**
- P0: shape-ID CLASSIFIER is INSENSITIVE (oracle~=random; shape identity robust to these morphisms) ->
  gate switched to EqM-lite which brackets clean x2 seeds (BASE 0.43 > RANDOM_DECOY 0.14 > RANDOM_VALID
  0.11 > ORACLE 0.077). All 3 anchors AUC 1.0. No hallucination (0D).
- P1 all 7 tasks: recovers true families (recall .78-1.0), avoids decoys (~0.00), valid (.82-.92), multi>
  single per-factor coverage, NO_ANCHOR degrades VALIDITY (anchor's role = on-manifold/decoy-avoid not
  coverage), impossible_control (BG unreachable) no hallucination.
- P2 EqM-lite: multi_independent DISCOVERED_MULTI gap 0.062 BEATS random_valid 0.107 BEATS oracle 0.078
  vs base 0.466; NO_ANCHOR 0.30 (anchor essential). **FIRST discovery>random in the whole arc.**

**Why it matters:** the recorded bottleneck ("operator architecture done; held-out TARGETING without labels
unsolved") is SOLVABLE when data has real hidden structure + a manifold anchor. Reopens the EqM bridge as a
viable bet (updates [[diff_eqm_symmetry_ladder]]).

**Honest caveats (do NOT overclaim):** synthetic gym + EqM-LITE proxy, NOT real EqM/FID. Classifier proxy
insensitive on non-shape-breaking factors. Payoff only where heldout shift is LARGE (single_rotation had
no headroom, base gap 0.004, all arms tie). discovered_multi TIES discovered_single on EqM-lite (one
morphism saturates field-robustness). FID NEVER auto-authorized -> RECOMMEND integration pending EXPLICIT
human approval.

**Phase 3 transfer (MNIST) PASS — 3 seeds (2026-06-07):** objects = real MNIST handwriting (colorized),
SAME machinery. multi task EqM-lite gap: DISCOVERED_MULTI 0.022-0.023 BEATS random 0.080 BEATS oracle 0.049
(base 0.117), NO_ANCHOR ~=base (anchor essential), decoy_use 0.00. Discovery pattern HOLDS on real images.
Phase 0->3 ALL PASS. (Perf note: train_eqm_lite/classifier precompute an aug BANK once, set
torch.set_num_threads(2); per-step eqm is CPU-thread-sensitive.)

**EqM bridge ATTEMPTED (2026-06-07) — smoke FAILS on real CIFAR, NOT submitted.** Built
`v14_multi_morphism_aug` (+ `_multi_morphism.py`, bridge150_v14 configs) porting the v17 recipe to CIFAR
EqM. Mandatory anchor/discovery smoke: the label-free anchor that hit AUC 1.0 on clean shapes/MNIST/dSprites
does NOT separate valid morphisms from decoys on natural CIFAR textures — random-conv ED accepts big_shear
(ED 0.10, lowest), in-domain AE accepts crop_erase/color_collapse, combined max(z) wrongly rejects valid
hue; each anchor leaks a different family. Discovery grabbed decoy big_shear (decoy_usage 0.96). Per
smoke-probe + 1-retune discipline NO 150ep GPU job submitted (would train EqM on object-destroying shear ->
CAFM-341-style disaster; smoke did its job). Root cause = anchor not symmetry-aligned on natural images
(documented risk realized). Postmortem `documentation/v17_bridge_smoke_postmortem.md`. v17 Phase0-3 positive
STANDS independently. Bridge needs a natural-image-robust label-free validity model (proxy-gate per-family
sep>0 BEFORE any GPU) OR fall back to known-symmetry (translate/crop) aug.

**Detector CRACKED (2026-06-07):** built natural-image-robust label-free detector
(`experiments/v17_cifar_detector.py`) = random-conv ED (catches crop_erase/color_collapse) + robust-AE
recon (AE trained on CIFAR + MILD valid nuisances flip/affine/brightness/hue -> vetoes big_shear, accepts
valid hue). Static per-family sep +0.378 (valid<all decoys). Wired into `_multi_morphism.discover(ae,
ae_weight)`; SUM combo needs ae_weight~50 so AE veto beats big_shear's low conv-ED -> decoy_usage 0.92(w8)
->0.157(w25)->0.079(w50,<0.10 gate, no decoy in top). `v14_multi_morphism_aug` defaults use_ae=true
ae_weight=50. **CIFAR bridge now PASSES decoy-avoidance smoke.** Remaining gate = human FID-approval to
submit 150ep 4-arm bridge (v00/vK/v14_disc/v14_rand), smoke `bridge_v14_smoke` (gpu_test 16-sample probe)
first, pre-reg gate discovered FID<random AND <=known-aug. Postmortem RESOLUTION section updated.

**CIFAR GYM — DISCOVERY REOPENED ON NATURAL IMAGES (2026-06-08).** Prior CIFAR bridge full-vs-full was a
WEAK negative: no visible->anchor gap, so any on-manifold op (incl big_shear) looked equal -> discovery
degenerated. New `experiments/v17_cifar_gym.py` builds REAL CIFAR gaps (5 tasks) + calibrate-before-payoff.
Findings: (1) TASK construction matters — `lowcolor_to_full` (desaturated visible vs full anchor) calibrates
clean (saturate Δ-0.30, decoys 0), `centerzoom` FAILS (info-destructive crop + big_shear games conv-ED),
position/no_rotation pass separability but NO payoff headroom (tiny gap, reflection artifacts -> oracle>base).
(2) OBJECTIVE matters — original `discover` reward `-max(convED, 50*AE_recon)` collapsed to `scale` on every
task and RAISED ED, because the robust-AE (nuisance set lacks saturation) mis-vetoes the correct family
`saturate` while accepting scale, AND on-manifold-ness != gap-direction. The move-hinge was INERT (move~3>>
margin 0.6; dropping it = no-op, verified). The 1 retune = `gap_aware=True` (bandit reward = -convED only,
AE downweighted to x5). RESULT 3 seeds: discovered=`saturate` {0.97,0.965,0.987}, decoy_usage {.008,.018,
.003}, payoff ED DISCOVERED {0.398,0.424,0.437} BEATS base {.656,.682,.719}, random_valid, random_decoy AND
oracle_valid {.584,.602,.644} every seed. **FIRST strong positive on NATURAL images.** Report
`documentation/v17_cifar_gym_report.md`. gap_aware flag added to `_multi_morphism.discover`. Best next: port
gap_aware reward into EqM bridge on a CONSTRUCTED CIFAR gap (desaturated train split vs full anchor), not
full-vs-full — thesis: discovery pays off only where useful symmetry isn't already known (crop). FID human-gated.

**EqM-LITE TRANSFER + REAL-EqM BRIDGE LAUNCHED (2026-06-08, user said "do all runs now, have ssh").**
(1) EqM-lite payoff added to gym (`--eqm_lite`): train TinyEqM on desat visible, eval eqm on full CIFAR.
3 seeds: DISCOVERED eqm_full {0.330,0.336,0.328} BEATS base {.356,.359,.353}, random_valid, random_decoy
AND oracle {.343,.347,.340} every seed -> the ED win TRANSFERS to a generative proxy, not just anchor
metric. (2) Built `v15_gap_morphism_aug.py` (diff-EqM): CONSTRUCTED-gap bridge — train EqM on DESATURATED
CIFAR (keep=0.25), FID ref=full CIFAR; arms base/known(crop)/random/discovered(gap_aware). Added
`m_saturate` to `_multi_morphism` VALID_FAMILIES (bridge had NO saturate -> discovery spread onto translate;
after add, picks saturate 0.87 decoy 0.002 on real desat CIFAR, verified). Pre-reg gate: discovered FID <
known < random < base (INVERSE of full-CIFAR bridge where crop won — here missing factor is COLOR, crop
can't restore it). Configs `configs/variants/bridge/gap15_*.json`. SUBMITTED to cluster (gpu, 150ep):
base 20086604, known 20086607, random 20093839, discovered 20093840 (+smoke confirmed sample path on 40G,
FID-eval OOMs on 20G gpu_test -> use gpu). Jobs RUNNING autonomously; epoch-50 FID ~3-4h, full ~overnight.
SSH session drops on 2FA -> re-bootstrap `scripts/cluster/ssh_bootstrap.sh` to poll. Report
`documentation/v17_cifar_gym_report.md` has all tables. SHAs: gym 9969a82, saturate-fix 23ac440.

**gap15 GATE PASS — REAL EqM (2026-06-08).** Constructed desaturation-gap bridge done (5K FID, seed0):
discovered(saturate) 37.16 < known-crop 40.59 < random 42.87 < base 43.50 = EXACT pre-reg ordering.
Discovery beats the strong human DEFAULT (crop) by 3.43 FID because crop can't restore chroma, discovered
saturate (decoy~0) can. FIRST real-EqM result where label-free discovery beats both random AND the
hand-designed default. Track A confirmed (gym->EqM-lite->real EqM all agree). NEXT: gap15 3-seed Welch
(discovered vs known seeds 1,2) for significance — NOT yet launched.

**FRAMING FIX (user, 2026-06-08):** crop/flip = STANDARD STRONG CIFAR DEFAULT, not oracle. RETRACT
"nothing to discover/crop near-optimal". Discovery wins if it BEATS the hand-designed lever (v10 ANM) OR
ADDS lift on top (hybrid). Staged success: min=disc<random; strong=disc<base~approach v10; major=disc<v10;
best=hybrid<v10.

**Track-B v10 ladder LAUNCHED overnight (2026-06-08), full CIFAR same protocol:** ladder150 arms v00
20244297, v10 20244298, v17rand 20244299, v17disc 20244300, hybrid 20244301 (gpu, SHA 9c02170). Built
`v16_hybrid_mine_morph` = v10 PGA mining + frozen discovered-morphism aug (different axes: v10 noised-space
robustness + v17 data-space on-manifold diversity). Pre-reg `documentation/v16_v10_comparison_ladder.md`.
Existing same-harness hints: crop 12.59 < v10 13.40 < v17disc 14.21 < v00 14.31 < v17rand 14.59 -> disc
LOSES to v10 alone, so HYBRID (complementarity) is the key arm; if disc loses run diagnostic. Generalized
"arbitrary symmetry" operator = NOT next (operator already solved in toy ladder; bottleneck is
targeting+validity not expressiveness; would lose interpretability) -> scoped exp(A)-as-extra-family on a
non-library-symmetry dataset is the right test IF library-discovery shows lift first.

**ASM (Adversarial Symmetry Mining) CPU ladder — NO GPU, gap15 stays flagship (2026-06-09).** Built
`asm_miner.py` (hard-positive symmetry mining: mine the valid transform HARDEST for current EqM via
Madry inner-max; EqM hardness = loss / COMMUTATOR ||F(Tx)-J_T F(x)|| / both; validity firewall anchor+AE+
decoy+mag+move) + `asm_cpu_ladder.py` (A/B/C) + `asm_decision.py` + `asm_diagnostic.py`. Gated CPU-decides-
GPU program. Results: CPU-A PASS (miner valid, decoys 100% rejected). CPU-B PASS (ASM_loss mines saturate
on desat gap, beats random — mechanism works WHERE THERE'S A GAP; but COMMUTATOR signal picked pad_crop
NOT saturate = underdelivered on lightly-trained probe). CPU-C FAIL 3-seed (SOLO ASM<random passed seed0
+0.016 but FAILED seeds1,2 -0.005/-0.001 = noise; base beats ALL aug every seed; static_v17 picked
color_collapse decoy). Per user rule "GPU only if SOLO 3/3" -> NO GPU. Adversarial objective did NOT
change the full-CIFAR no-gap outcome (same as static v17 ties-random). Verdict `documentation/
asm_cpu_verdict.md`. ASM machinery preserved/reusable for future REAL-gap tasks. Lesson: discovery value
(static OR adversarial) is GAP-CONDITIONAL; full CIFAR has no gap. Gated CPU caught noise at ~0 GPU cost.

**OPEN: cluster 3-seed jobs 20651699-706 (gap15 disc/known s1,2 flagship Welch + ladder v10/hybrid s1,2)
still running, polling blocked on 2FA — re-bootstrap to fetch finals + Welch.**

**HPSM EQUIVARIANCE-CONSISTENCY — FAILS AT REAL SCALE, TinyEqM PROXY INVERTED (2026-06-13).** Built HPSM
(hard-positive symmetry mining, dual of v10 ANM) + general task-agnostic version (`F(x_t+gd)=sg(F)-d` from
EqM target, no named symmetry/J_T) + v18/v19 variants + 9-arm GPU ladder. TinyEqM CPU proxy said
consistency term is the lever (random+consist≈HPSM beat all, +0.10). REAL EqM-B 150ep 5K FID INVERTED it:
randcons(random+commutator-consistency) 32.77 CATASTROPHIC > base 14.31 > general(EqM-derived consist)
14.15 ~ v10 14.15 > randnocons(plain random aug) 13.31. Named commutator (hand-coded J_T) HARMFUL at scale
(likely wrong-sign J_T / conflicts EqM obj); general consist NEUTRAL; only plain aug mildly helps (not
novel). PRIMARY win condition fails hard. named/hybrid arms timed out (online K8 mining too slow 20h) but
moot. LESSON: validate equivariance/consistency regularizers on REAL-capacity model — underfit tiny proxies
REWARD regularization a properly-trained model is HURT by (proxy didn't just overstate, it flipped sign).
Infra: jobs died exit53 home-quota-full (deleted old final.pt/checkpoint.pt to free; preserve IN-1K
baselines); v19 hybrid OOM bs128 (ANM+HPSM+commutator ~4 graphs>79GB)->bs64. Code preserved
(hpsm_miner/v18/v19/hpsm_ladder) but thesis dead. **gap15 static gap-aware discovery (+3.4 FID real EqM)
remains the ONLY real win across the whole arc.** Verdict `documentation/hpsm_theory_and_ladder.md`.

**Next (gated, user to decide):** (a) confirm on dSprites/Shapes3D (native GT factors, non-MNIST); (b)
write up the Phase 0-3 positive as the contribution; (c) real-EqM bridge (cluster 2FA
`scripts/cluster/ssh_bootstrap.sh`) ONLY on explicit human approval — FID never auto. State in
`.state/pipeline.json` (needs_user_input set), write-ups in `results/summary.md` + `results/v17_report.md`,
bridge update `documentation/eqm-bridge-plan.md`.
