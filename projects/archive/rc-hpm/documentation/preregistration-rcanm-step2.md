# Pre-registration — RC-ANM Step 2: CIFAR-mini certified-training ladder (GPU)

Written 2026-06-13 BEFORE Step 2 runs. Gated on Step 1 (P1 or P2). Tests
whether the certification BUYS something at a scale where ANM utility is real
(the toy could not — fixed_anm damaged it though v10 helps IN-1K).

## Arms (training, equal step budget; FID 5k quick signal)
1. vanilla          no mining.
2. fixed_anm (v10)  fixed eps_ball=0.5, uncertified (the scale-proven mechanism).
3. aggressive_anm   eps_ball=1.5, uncertified (damage control).
4. rc_anm           per-step accept/reject mined endpoints by the EqM-native
                    safety score validated in Step 1 (the score that separated
                    safe/unsafe gradients); reject -> un-mined endpoint.
   (If Step 1 = P2: RC-ANM uses the aggressive eps with the filter — "mine
    aggressively, safely".)

## Model / data
EqM-B/2 or S/2 on CIFAR-10 (32x32, latent via VAE or pixel per the EqM-restart
CIFAR path), equal MAX_STEPS all arms, single GPU. Teacher = EMA snapshot
(richer than the 2D toy — the R2' fix). 3 seeds if budget allows, else 1 +
documented.

## Gate G-RCANM-step2
(a) PREMISE-AT-SCALE: aggressive_anm worse than vanilla (FID).
(b) v10 REPRO: fixed_anm <= vanilla (mining helps at this scale, as IN-1K).
    If (b) fails, CIFAR-mini is still too small for ANM utility -> escalate to
    B/2 IN-1K (the only proven-utility scale), human-gated.
(c) CERTIFICATION BUYS: rc_anm <= fixed_anm (certified filter matches/beats
    blind v10) OR (P2 path) rc_anm < vanilla while aggressive_anm > vanilla
    (filter unlocks safe-aggressive gains v10 leaves on the table).
(d) realized safety risk <= alpha.
Branches:
  W1 rc_anm beats/matches v10 + safety holds -> RC-ANM validated at scale ->
     write up; propose B/2 IN-1K confirmation (human-gated).
  W2 v10 helps but rc_anm only ties -> "certified v10, no FID cost" (safety
     contribution stands, utility neutral).
  W3 (b) fails -> CIFAR too small -> escalate to IN-1K.
  W4 rc_anm worse than v10 -> filter hurts -> postmortem.

## Discipline
Reuse rc-hpm EqM + transport (mine_negative) + the Step-1 safety score. Job
tracked in pipeline.json active_runs. FID never auto-promoted to a scale claim;
B/2 IN-1K only after W1 + explicit human approval.
