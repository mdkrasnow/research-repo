# Postmortem — G2: EqM bridge fails at toy scale (2026-06-12)

## Verdict
Both Stage-1 arms FAIL G2(c) on their pre-registered primaries
(results/e1_12_verdict.json, 5 seeds each, margins = 2x vanilla seed-SD):

| arm | primary | value | vanilla | margin | pass |
|---|---|---|---|---|---|
| disp_uniform | field MSE | 1.104 | 0.837 | 0.086 | no (worse) |
| rc_hpm | field MSE | 0.838 | 0.837 | 0.086 | no (tie) |
| rc_repulsive | field MSE | 0.849 | 0.837 | 0.086 | no (tie) |
| **oracle_pairs (+)** | field MSE | **0.890** | 0.837 | 0.086 | **no (worse)** |
| anm_cert | recall dist | 0.999 | 0.801 | 0.647 | no (worse, in noise) |
| anm_live_neg (-) | recall dist | 1.145 | 0.801 | 0.647 | d_damages: no (in noise) |

## The load-bearing finding: the POSITIVE control is null
oracle_pairs = SupCon with TRUE labels on the field's penultimate activations.
If perfect pair information does not improve field accuracy, no certified
mining of pair information can. The entire ARM A mechanism class
(contrastive aux on activations -> better EqM field) has no effect channel in
this toy. This is a mechanism finding, not a tuning failure:
- aux/base ratios healthy (0.02-0.21, non-saturating) -> the G2 retune trigger
  ("(b) saturates -> mechanism fights c(gamma)") did NOT fire; no retune is
  pre-registered for an oracle-null failure mode; none taken.
- RC-HPM risk monitoring during training stayed within certified alpha
  (per-run logs in results/e1_12_results.json) — the certification machinery
  worked; the mechanism it certifies just has nothing to add here.

## ARM B reading
anm_cert calibrated (eps_ball=0.1, flip-risk controlled) but recall-distance
is dominated by rare-mode dropping noise (vanilla seed-SD ~0.32 on mean 0.80).
The damage arm (live-student, eps x10) lands worse on mean (+0.34) but inside
the margin. The 2D toy's rare-mode metric is too noisy to bracket either
direction at n=5. Inconclusive, not exculpatory.

## Scope limits (do not over-conclude)
- 2D MLP field, 8-mode mixture. Dispersive Loss (Wang et al.) reports gains
  at ImageNet scale with transformer backbones — representation-level
  regularization may need representation-scale models. The toy answers
  "does the bridge show up at CPU scale": NO. It cannot answer "is the bridge
  dead at B/2 scale".
- Per the tree, any Stage-2 (GPU) attempt now requires new toy-scale evidence
  or an explicitly narrowed harm-bounding story.

## What survives (per pre-registered branches)
1. STATISTICAL MACHINERY VALIDATED: G-1 (9/9 bug injections detected),
   G0 (20/20 seeds certified at alpha=0.1, 0 exceedances, vacuous-on-random).
   The LTT/HB certified-mining pipeline is correct and portable.
2. HARM-BOUNDING STORY (G1): naive hard mining craters linear-probe accuracy
   (0.27 vs 0.95); RC-HPM mines under certified risk with NO damage. Utility
   gains over no-mine/RINCE: none at toy scale (probe at ceiling).
3. PREMISE (G1.5): on CIFAR with a frozen rn18 teacher, gradient-weighted
   damage density concentrates 14.3x in the top hardness decile (3x required),
   decaying with gamma exactly as the P4 gamma-conditional design assumed.
4. The contrastive-CL standalone track (certification as safety layer for
   hard-pair mining in contrastive learning) survives per the tree; the
   EqM-bridge specific claim does not, at toy scale.

## Stop-condition invoked
CLAUDE.md research rules: "Experiment only adds complexity without testing a
clear mechanism" + tree branch "EqM bridge weak". Killed rather than retuned;
the single G2 retune budget remains UNSPENT (no pre-registered trigger fired).
