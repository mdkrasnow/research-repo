---
name: rc-hpm-cpu-tree
description: "rc-hpm CPU decision-tree complete — machinery validated, harm-bounding framing, EqM bridge dead at toy scale (oracle-null)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 62c875aa-4edb-425f-a84e-44e4b9b8705b
---

rc-hpm (projects/rc-hpm, EqM fork + Risk-Controlled Hard-Pair Mining): CPU
portion of decision tree (documentation/experiment-decision-tree.md v2) ran
end-to-end 2026-06-12. Gates: G-1 PASS (9/9 bug injections detected),
G0 PASS (20/20 seeds certified α=0.1, 0 exceedances), G1 K2-only (naive
hard-mine craters probe 0.27 vs 0.95; RC prevents at certified risk; NO
utility gain → harm-bounding+guarantee framing, pre-registered branch),
G1.5 PASS (CIFAR damage density 14.3× at hard tail), G2 FAIL both arms —
LOAD-BEARING: oracle (true-label) contrastive aux on activations does NOT
improve 2D EqM field (0.890 vs 0.837 MSE) → bounds the whole arm-A class;
E1.3 skipped (entry condition). Postmortem:
documentation/postmortem-g2-eqm-bridge.md. Amendments A1–A4 all pre-gate
(calibration arithmetic m=250/α₀=0.10; data-flow + multi-stat detectors;
8-unequal-mode task; γ-window + flip risk).

**Why:** future sessions must not re-litigate: bridge-at-toy-scale answered
NEGATIVE via positive-control bound, not via tuning failure; the LTT/HB
certified-mining library (rc_hpm/core.py) is validated and portable.

**How to apply:** Stage 2 (GPU) NOT entered — needs_user_input in
projects/rc-hpm/.state/pipeline.json offers (a) contrastive-CL standalone
harm-bounding track, (b) bridge-toy redesign, (c) park. Don't scale anything
that didn't pass G2. Related: [[diff_eqm_v10_in1k_3seed]] (v10 mining DOES
work on EqM at IN-1K — different mechanism: endpoint PGD on residual, not
contrastive aux).

## D2 update (2026-06-12) — band hypothesis FALSIFIED, hardness HARMS
D2 difficulty-ladder ran (documentation/d2-verdict-postmortem.md). Branch B2:
RC-HPM hard-pair-mining utility RETIRED at CPU scale. DECISIVE F15 decomposition
at alpha=0.40 (where certified-hard supply is non-empty): rc_hpm HARMS no_mine
by 6-24 pts, but cert_random_k (certified, NO hardness) ~= no_mine. So hardness
selection harms; certification is safe. Mechanism: hardness and certifiability
ANTICORRELATED (E1.0 14.3x) — hardest pairs = residual false-negatives the gate
can't clean, mining them = FN-push damage. NO winning alpha (tight->degenerate,
loose->harm). H-B' HOLDS (B3 not fired, certify-neg-only 95% protection, FP-pull
minor). RINCE foil DEAD. gamma kNN-Laplacian probe validated 3/4.
**Why:** the "hard-pair mining" framing itself is falsified; the live axis is
certification/curriculum, NOT hardness. Next = certified-pair-curriculum (no
hardness) or debiased-CL-minimal — NEW hypothesis/prereg.
**Process win:** pre-registered supply-vs-alpha LINCHPIN overturned a premature
B2-structural call (supply was throttle-measured at tight alpha, not absent).
Deviations D3 (safety arms on supply rungs), D4 (linchpin refocus), D5 (band at
4*alpha_0). Awaiting human: (a) prereg certification-curriculum exp, (b) write
safety+guarantee paper, (c) park.

## PIVOT -> RC-ANM (2026-06-13)
Mainline pivoted from contrastive-EqM (archived negative) to RC-ANM (Risk-
Controlled Adversarial Negative Mining for EqM): certify the MINED ENDPOINT
(v10/ANM lineage), not contrastive pairs. Docs: pivot-rc-anm.md,
preregistration-rc-anm{,-v2}.md, postmortem-rcanm-r4.md. Code: rc_hpm/rc_anm.py
(PGD candidates + teacher safety scores r_basin/field/target/inflate/return +
per-gamma LTT-certified eps_ball), experiments/rcanm_ladder.py.
D3 contrastive ARCHIVE: C2 (cert_random_k = cheap safe regularizer +0.01-0.02,
no curriculum gain) - contrastive path closed.
RC-ANM toy ladder: R4 then R2'. Concept VALIDATED (oracle-safe 0.877 ~ vanilla
0.837 + best coverage; dose-matched control oracle42 0.877 vs random42 1.285 ->
basin-safety SELECTION real, not dose). Premise VALIDATED (aggressive ANM
damages 2.67 vs 0.84). BUT toy teacher field too collapsed: basin proxy ~chance
vs analytic oracle (bal acc 0.49@1000/0.53@2500 steps) -> functional NOT
CPU-instrumentable. fixed_anm damages toy (2.67) though v10 HELPS IN-1K -> 2D
toy is WRONG INSTRUMENT for ANM utility AND for the teacher basin proxy.
**Why:** RC-ANM channel EXISTS (unlike contrastive oracle-null) - the pivot is
live, but must move to CIFAR (rich teacher) to instrument it. Devs D3-D6.
**Next (human-gated):** CIFAR-mini RC-ANM (vanilla/v10-fixed-ANM/RC-ANM,
rn18-or-EqM-EMA teacher). needs_user_input set. Don't fish for a toy proxy
(prereg forbids). GPU/FID never auto.

## RC-ANM Step-1 SCALE PROBE (2026-06-13, GPU job 22507558) -> P3 NON-PROBLEM
Ran premise probe on trained v10 EqM-B/2 IN-1K ckpt (holylabs), inference-only,
eps sweep. unsafe-fraction rises with eps (0.05->0.64 field, inflate 3->12x) BUT
unsafe endpoints do NOT poison the EqM training gradient at ANY eps (grad_cos
unsafe 0.047 vs safe 0.064, Welch p=0.40, n=256). Pre-registered branch P3 ->
STOP before Step-2 training (would certify a non-problem).
**Structural finding (publishable):** EqM target (x1-eps_adv)*c(gamma) references
the REAL x1, so adversarial ENDPOINT mining can't corrupt the data-direction ->
ANM helps EqM UNCERTIFIED (=why v10 works: FID 27.58 vs 31.41) and certification
has nothing to bound. Adversarial mining needs risk-control ONLY when the
objective lacks a fixed real anchor: contrastive YES (D2 damage real), EqM
regression NO. Third convergent scale confirming "risk-control finds no problem
where the base method works."
**Cluster ops:** GitHub push of filter-repo-rewritten history (751MB, after
purging accidental 350MB data/ commit) too large/flaky over HTTPS+SSH; GPU work
ran via rsync to /n/home03/mkrasnow/research-repo (no-clone sbatch
projects/rc-hpm/slurm/jobs/rcanm_step1_local.sbatch). v10 ckpts:
/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/imagenet1k_80ep_v10_b2_lambda{03_seed2,10_seed0}/.../final.pt;
IN data /n/holylabs/ydu_lab/Lab/raywang4/imagenet/train. SSH master expires ->
needs `! scripts/cluster/ssh_bootstrap.sh` (2FA).
**Status:** all NEEDED GPU jobs ran; P3 STOP = no more warranted. needs_user_input:
(a) write up insight+negatives, (b) pivot risk-control to contrastive-CL (has
poisonable target), (c) park.
