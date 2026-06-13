# D2 verdict + postmortem — H-A′ band hypothesis (2026-06-12)

Branch (pre-registered): **B2 (utility claim retired) for RC-HPM**, with a
strong mechanistic redirect the next hypothesis must absorb. B3 did NOT fire
(H-B′ holds). Result is sharper than B2's letter ("gap ≈ 0"): the gap is
robustly NEGATIVE, and the cause is decomposed.

## Headline finding: hardness and certifiability are anticorrelated, so
## hardness-mining net-HARMS; certification-without-hardness is the safe point

At α = 0.40 (the P2 4α₀ endpoint, the only budget where certified-hard supply
is non-empty on high-headroom rungs), per-rung probe accuracy (10 seeds):

| rung | ρ_tail | H | S@.40 | no_mine | SupCon⊕ | rc_hpm | cert_rand_k | rince | naive⊖ |
|---|---|---|---|---|---|---|---|---|---|
| K20_s2.4_a0.0 | 0.21 | .063 | .92 | .749 | .812 | **.507** | .774 | .766 | .344 |
| K40_s2.4_a0.8 | 0.19 | .060 | .88 | .647 | .707 | **.469** | .670 | .664 | .339 |
| K10_s2.4_a0.0 | 0.40 | .052 | .66 | .828 | .880 | **.718** | .848 | .839 | .280 |
| K20_s2.0_a0.8 | 0.38 | .040 | .64 | .862 | .902 | **.715** | .874 | .876 | .376 |
| K40_s2.4_a1.6 | 0.59 | .047 | .40 | .784 | .831 | **.722** | .803 | .798 | .500 |

Three facts, every rung:
1. **rc_hpm < no_mine** by 0.06–0.24 (gap robustly negative; C1–C4 all fail;
   B1 band NOT found).
2. **cert_random_k ≈ or > no_mine** (+0.01 to +0.02) — certification alone is
   safe-to-mildly-positive.
3. **rc_hpm < cert_random_k** by 0.08–0.27 — hardness selection, NOT
   certification, is what harms.
4. rc_hpm > naive⊖ everywhere (+0.13 to +0.44) — certification DOES protect
   relative to uncertified mining; the harm is net vs the no-mine baseline,
   not vs naive.

### Mechanism (ties to E1.0's 14.3× premise — the tension is the point)
E1.0: pair-label damage concentrates 14.3× in the top similarity decile, i.e.
ρ̂(s) is high exactly where s is high. Hardness-mining SEEKS high s. Therefore
the hardest certified negatives are disproportionately the residual
false-negatives that survived the gate — pushing them apart IS the FN-push
damage (D1-#3). Hardness fights certifiability: it targets exactly the region
the gate can least clean. Certification-without-hardness (random among
certified, or a confidence curriculum) avoids the residual-FN tail and is
neutral-to-positive. **The "hard-pair mining" framing is falsified; the live
operating point is "certified-pair curriculum, no hardness."**

This UPGRADES D1-#7 ("cert_random_k ≈ rc_hpm; hardness adds nothing"): at the
α where supply is real, cert_random_k > rc_hpm decisively — hardness does not
merely fail to help, it HARMS.

### The risk dial is real but one-sided (no winning α)
- tight α (0.10): certified-hard supply → 0; rc_hpm degenerates to no_mine
  (D1: gap ≈ 0, no harm, no gain).
- loose α (0.40): supply abundant, but the admitted certified-hard set carries
  enough residual FN that mining it net-harms.
There is NO α at which rc_hpm > no_mine. The frontier runs from "no effect"
(tight) to "harm" (loose). The dial controls how much you lose, not whether
you win.

### Unexploited label signal
SupCon⊕ beats no_mine by 0.04–0.06 (real headroom exists). No unsupervised arm
captures it: cert_random_k gets at most +0.02. The label information that would
help is NOT recoverable by hardness-mining OR random certification at these
budgets — an open question, not a closed one.

## B3 (H-B′): did NOT fire — full gate is NOT load-bearing
On the supply-bearing damage rung (K40_s1.2_a0.8, naive damage 0.395):
retention = (rc_neg_only − naive)/(rc − naive) = (0.942−0.589)/(0.960−0.589)
= **0.951 ≥ 0.80**. Certify-negatives-only keeps 95% of full RC's protection
at half the machinery/label cost. FP-pull probe (closes D1-#3 vacuity):
naive_pos 0.951 vs no_mine 0.984 → uncertified positives harm only 0.033 (vs
FN-push 0.395), so dropping mined positives is safe. The positives channel is
not load-bearing.

## RINCE foil: DEAD (no separation)
Concentrated boundary-crossing view noise: rc − rince = −0.008 (RC not better);
diffuse similar. The silent-vs-visible-failure figure does not materialize in
this generator. The safety claim rests on D1's 3/3 visible ABORT behavior, not
the RINCE contrast. (Pre-registered as a possible outcome; honestly retired.)

## γ probe: re-instrumented and VALIDATED
kNN symmetric-normalized-Laplacian eigengap recovered true K on 3/4 known-K
rungs (K̂=19 at K=20). D1-#8's 1e-15 artifact resolved. γ available as a
covariate for any future theory-linked analysis (not used as a gate here).

## Process note — the linchpin worked exactly as designed
A premature "B2 / band absent" lean (from the pre-flight's H-S anticorrelation)
was OVERTURNED by the pre-registered supply-vs-α linchpin: supply was
throttle-measured at α₀=0.10, not structurally absent (Dev D5). Without the
linchpin this project would have filed a false structural negative. The
band genuinely does not exist — but for the deeper hardness-harms reason above,
established only after the linchpin forced the α-correct measurement.

## What carries to the next hypothesis (escalation)
The hardness axis is DEAD (harmful). Two live directions, both CPU-testable:
1. **Certified-pair curriculum (no hardness):** cert_random_k is safe and mildly
   positive; formalize as confidence-ordered (not similarity-ordered) inclusion.
   Does an explicit confidence curriculum beat random-certified? Recover more of
   the SupCon headroom?
2. **Debiased-contrastive minimal form:** B3 + FP-pull say the method reduces to
   "down-weight high-same-class-probability pairs in the denominator, certified,
   negatives-only." That is a debiased-CL variant with a risk dial — the safety
   product, stripped of hardness and positives.
Both keep the validated machinery (rc_hpm/core.py); neither mines hard pairs.

## EqM relevance (unchanged)
Arm A (contrastive-on-activations) stays dead (D1-#1b, headroom-proven). EqM
endpoint mining (v10) lives at scale via diff-EqM (FID 27.58). Nothing in D2
reopens arm A. The D2 finding — hardness-mining concentrates on the residual-FN
tail — is a caution for ANY hardness-based scheme on EqM, including v10:
v10 mines hard ENDPOINTS (PGD on residual), a different object than contrastive
pairs, but the "hardest = least certifiable" tension is worth checking there
before certifying v10 (the standing GPU question).

## Branch declaration
PRIMARY (gap RC−no_mine regression): C1=C2=C3=C4 = False → **B1 band NOT found**.
gaps robustly negative (B2 destination reached, with mechanism). **B2: utility
claim for RC-HPM retired at CPU scale.** B3: not fired (H-B′ holds, full gate
not load-bearing). No improvised branch — the redirect (certification-curriculum,
not hardness) is a NEW hypothesis for a NEW pre-registration, not a D2 rescue.
