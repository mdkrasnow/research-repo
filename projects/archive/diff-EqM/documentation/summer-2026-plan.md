# Summer 2026 Experiment Plan — diff-EqM toward Publication (Branch B-Both)

**Created**: 2026-05-19. Revised after Lin et al. discovery: pivoted to Branch B-Both (PGD-on-target + CAFM-style discriminator AT; primary on EqM, secondary head-to-head on SiT).
**Horizon**: May 19 – Sept 1, 2026 active. Write-up through Oct 1.

## Targets

| Venue | Deadline | Status |
|---|---|---|
| NeurIPS 2026 workshop | 2026-08-29 | PRIMARY derisk path (~14 wks) |
| ICLR 2027 main | ~2026-10-01 | PRIMARY full-paper target (~19 wks) |
| NeurIPS 2026 main | 2026-05-06 | MISSED |
| ICML 2026 | 2026-01-28 | MISSED |
| ICML 2027 | ~2027-01-28 | Safety net |

## Sharpened claim (locked 2026-05-19; revised post-Round-2 lit review)

**"First adaptive hard-negative mining for regression-target generative models. PGD-mined hard examples on the velocity-field regression target (v10) compose with CAFM-style discriminator post-training (Lin et al. 2026) to compound FID gains on Equilibrium Matching (Wang & Du 2025). Two losses attack complementary failure modes: discriminator catches global distributional mismatch; PGD-mining catches local regression failure."**

**Direct external validation**: VeCoR (Hong et al. 2025, arxiv 2511.18942) §7 explicitly lists "**adaptive hard-negative mining**" as open future work for velocity contrastive regularization. v10 is exactly that.

Two contributions:
1. **v10 alone**: first adaptive (adversarial) hard-negative mining on the regression target. Distinguishes from VeCoR's heuristic augmentation negatives and from DAT's discriminative BCE negatives.
2. **v10 + CAFM**: first composition of PGD-input-mining with GAN-discriminator AT on regression-target flow / EqM models.

Third (Phase 5): SiT head-to-head confirms transfer to standard flow matching.

Distinct from:
- Lin AFM/CAFM (no PGD on input).
- Wu DAT (PGD on classifier-as-EBM BCE loss; classical EBM, not flow/EqM).
- Geng AEBM-Diff (generator-discriminator embedded in diffusion steps; not PGD-on-input, not EqM).
- Kong ACT (adversarial consistency models; no PGD).
- Rob-GAN (defensive classifier, not generative SOTA).

## Pre-registered gates

| Phase | Gate | Action on fail |
|---|---|---|
| 0 | Lit synthesis written. CAFM repo cloned + smoke-tested on small SiT-B/2 sample. | Block Phase 1 until CAFM mechanics understood. |
| 1a | CAFM port to EqM-B/2 runs end-to-end. No NaN. Loss curves sensible. | Diagnose port; if intractable → fall back to B-SiT (run everything on SiT). |
| 1b | CAFM-on-EqM-B/2 alone (10 ep post-training) achieves FID < 25 (vs vanilla 31.41). | Investigate; if CAFM doesn't help EqM at all → strong negative result, single-paper finding; pivot story. |
| 2 | v10 + CAFM combined beats CAFM-alone by ≥0.3 FID at seed 0. | 1 retune of λ_v10 ∈ {0.03, 0.3, 1.0}. Then revert to v10-only workshop paper. |
| 3 | 3-seed v10+CAFM mean beats CAFM-alone mean by ≥0.5 FID at p<0.10. Plus scaling on EqM-S/2 IN-100. | Workshop only with single-seed result; ICLR claim narrows. |
| 4 | Workshop draft ready Aug 22 | Slip workshop; focus ICLR. |
| 5 | SiT head-to-head: combination outperforms CAFM-alone by ≥0.3 FID on SiT-B/2 IN-256. | Acknowledge as EqM-specific; ICLR claim narrows. |

## Phase 0 — Weeks 1–3 (May 19 – Jun 9): Lit lock + CAFM verify

Extended to 3 weeks for full lit review + CAFM repo reproduction.

| Task | Output | GPU-h |
|---|---|---|
| 0.A Lit review (~18 papers incl. AFM/CAFM/ACT/DAT/Geng/EqM full reads) | `literature/*.md` + `SYNTHESIS.md` | 0 |
| 0.B Apply synthesis to plan + CLAUDE.md + memo | committed edits | 0 |
| 0.C Clone Lin repo; reproduce CAFM-on-SiT-B/2 CIFAR or small IN-256 smoke | reproduction note | ~40 |
| 0.1 v02 IN-1K cancellation postmortem | `debugging.md` | 0 |
| 0.2 Port plan: CAFM mechanics → EqM-B/2 design doc | `cafm-eqm-port-design.md` | 0 |
| 0.3 v10 + CAFM combined loss design doc | `v10-cafm-combination-design.md` | 0 |
| 0.4 v11/v12 fallback sketches | `v11_fallback_sketch.md` | 0 |
| 0.5 Arxiv weekly sweep started | `literature/arxiv-weekly-sweep.md` | 0 |

**Exit gate**: SYNTHESIS.md written + CAFM smoke passes + port design doc reviewed.

## Phase 1a — Weeks 4–5 (Jun 9 – Jun 23): CAFM port to EqM

Port CAFM machinery into EqM training codebase.

| Task | GPU-h |
|---|---|
| Implement JVP-based discriminator (DiT-style with [CLS]) for EqM | 0 |
| Implement least-squares GAN loss + centering penalty + (no R1/R2 since CAFM uses continuous formulation) | 0 |
| Adapt N=16 discriminator updates per generator update to EqM time conditioning | 0 |
| End-to-end smoke: 1 epoch CAFM post-training of vanilla EqM-B/2 IN-1K | ~30 |
| Validate diagnostics: discriminator loss, OT loss, FID trajectory | 0 |

**Gate**: Runs to completion. No NaN. Discriminator loss not collapsing.

## Phase 1b — Weeks 6–7 (Jun 23 – Jul 7): CAFM-alone EqM baseline

Post-train trusted vanilla EqM-B/2 80ep (FID 31.41) with CAFM 10 ep.

| Task | GPU-h |
|---|---|
| CAFM-on-EqM-B/2 seed 0, 10 ep post-training | ~150 |
| FID eval 50K samples, guidance-free + guided | ~10 |

**Gate (1b)**: FID < 25 (vs 31.41). Strong gain confirms CAFM-to-EqM port works.

## Phase 2 — Weeks 8–9 (Jul 7 – Jul 21): v10 + CAFM combined

Implement combined loss:
```
L_total = L_CAFM_gen(G, D) + λ_v10 · L_base(x_t + δ*)
δ* = argmax_{||δ||≤ε} L_base(x_t + δ)
```

| Task | GPU-h |
|---|---|
| Implement v10 PGD step inside CAFM training loop (every Nth gen update) | 0 |
| Seed 0 of v10+CAFM-on-EqM-B/2 | ~200 |
| Diagnostics: L_base, L_hard, L_disc, ||δ||, ratio, FID | 0 |

**Gate (Phase 2 exit)**: v10+CAFM seed 0 FID beats Phase 1b CAFM-only FID by ≥0.3.

## Phase 3 — Weeks 10–12 (Jul 21 – Aug 11): 3-seed ablation + EqM-S/2 scaling

Ablation table:

| Condition | EqM-B/2 IN-1K seeds | EqM-S/2 IN-100 seeds |
|---|---|---|
| Vanilla EqM (trusted) | 1 (existing) | 3 (new) |
| v10-only (no CAFM) | 1 | 1 |
| CAFM-only | 3 (from Phase 1b + 2 new) | 3 |
| v10 + CAFM | 3 | 3 |

| Task | GPU-h |
|---|---|
| EqM-B/2 IN-1K: CAFM-only seeds 1, 2 | ~300 |
| EqM-B/2 IN-1K: v10+CAFM seeds 1, 2 | ~400 |
| EqM-B/2 IN-1K: v10-only seed 0 (sanity) | ~150 |
| EqM-S/2 IN-100 baseline + 3 conditions × 3 seeds | ~300 |
| FID eval all conditions | ~50 |

**Gate (Phase 3 exit)**: v10+CAFM 3-seed mean beats CAFM-only 3-seed mean by ≥0.5 FID with Welch t p<0.10.

## Phase 4 — Workshop write-up (Aug 11 – Aug 29)

Workshop paper headline:
- CAFM-to-EqM port (first application).
- v10+CAFM combination beats CAFM-alone.
- Ablation table EqM-B/2 IN-1K (3 seeds each, 4 conditions).
- Scaling sketch on EqM-S/2 IN-100.

Submit by **Aug 29**.

## Phase 5 — SiT head-to-head + ICLR (Aug 29 – Oct 1)

Add SiT-B/2 IN-256 head-to-head:
- Reproduce Lin's CAFM-on-SiT-XL/2 single seed (sanity vs paper numbers).
- v10+CAFM on SiT-B/2 (smaller for compute) 3 seeds.
- Compare directly to Lin's published numbers.

| Task | GPU-h |
|---|---|
| SiT-B/2 IN-256 reproduce CAFM-only (seed 0) | ~200 |
| SiT-B/2 IN-256 v10+CAFM 3 seeds | ~400 |
| FID eval | ~30 |

**Gate (Phase 5)**: v10+CAFM on SiT-B/2 beats CAFM-only by ≥0.3 FID. Confirms claim generalizes beyond EqM.

ICLR submission: Oct 1.

## Compute budget

| Phase | GPU-h |
|---|---|
| 0 | ~40 |
| 1a | ~30 |
| 1b | ~160 |
| 2 | ~200 |
| 3 | ~1200 |
| 4 | 0 |
| 5 | ~630 |
| Buffer (15%) | ~330 |
| **Total** | **~2600** |

Tight on 2400 GPU-h budget. Mitigation: drop v10-only Phase 3 seed (saves 150) if needed; reduce SiT Phase 5 seeds 3→2 (saves 130).

## Risks (updated 2026-05-20 post Phase 0.3 PASS)

| Risk | Likelihood | Severity | Mitigation | Status |
|---|---|---|---|---|
| v10 mechanism collapses (Briglia invariance threat) | ~~M~~ → **L** | Plan blocker | v11 equivariant fallback ready | **RETIRED** — v10 trained 150 epochs CIFAR without collapse, ratio stable 1.047-1.049 |
| v10 saturates on EqM-B/2 (like v02 cosine) | ~~M~~ → **L** | Workshop kill | Mechanism design uses L2 regression (unbounded gradient) | **RETIRED** — non-saturating signature confirmed in 150-epoch CIFAR run |
| CAFM port to EqM fails (time conditioning incompatible) | M | Plan blocker | Phase 1a gate; fallback B-SiT (all on SiT) | Awaiting smoke 13997995 result |
| v10 + CAFM don't compound (redundant signals) | M | Workshop only | Phase 2 gate forces decision early | Awaiting Phase 1b/2 |
| Compute overrun | M | Slip schedule | Drop seeds; smaller scaling phase | Tracking |
| Lin v3 paper adds PGD-on-input before Oct 1 | M | Scoop | Weekly arxiv sweep; pre-register via workshop Aug 29 | Sweep 2026-05-20: no new HIGH threats |
| Du/Wang adversarial-EqM follow-up before Oct 1 | M | Scoop | Same | Sweep 2026-05-20: no new HIGH threats; EqM paper own "future work" cite supports our direction |
| EqM reviewer pushback ("why not just SiT?") | L | Claim narrows | Phase 5 SiT head-to-head answers this | Phase 5 plan locked |
| Discriminator training instability | M | Lost time | Lin's recipe (LSGAN + N=16 + low LR) is published stable | Awaiting Phase 1b |
| Cluster SSH credentials drop | M | Day-scale delay | Refresh via login when notified | Recurring; observed 2x in past 24h |

**Updated probability estimates** (post Phase 0.3 PASS):

| Outcome | Pre-0.3 | Post-0.3 |
|---|---|---|
| v10 mechanism sound | 60% | **85%** |
| v10 alone beats vanilla EqM-B/2 IN-1K | 40% | **55%** |
| CAFM port to EqM works at all | 50% | 50% (unchanged; different mechanism) |
| v10+CAFM compounds (Phase 2 gate PASS) | 50% | 50% (unchanged) |
| Workshop paper publishable | 60% | **70%** |
| ICLR main publishable | 35% | **40%** |

## Hard rules (per CLAUDE.md)

1. One change per experiment.
2. Re-baseline each scale.
3. Pre-register kill conditions.
4. Max 1 retune per failing direction.
5. PI updates on every gate pass/fail.
6. Weekly arxiv sweep, Mondays.

## Decisions locked

1. **Branch B-Both**: PGD + CAFM combination, primary on EqM, secondary on SiT.
2. **Venues**: NeurIPS 2026 workshop + ICLR 2027.
3. **Code base**: clone + adapt Lin's MIT-licensed CAFM (github.com/ByteDance-Seed/Adversarial-Flow-Models).
4. **Compute**: 2400 GPU-h budget approved; revised total ~2600 with cushion.
5. **Lit review leads Phase 0**: SYNTHESIS.md gates code work.

## Open

- v02 IN-1K cancellation root cause.
- v11/v12 fallback variants (if Phase 2 gate fails).
- CAFM mechanics may need adaptation for EqM's c(γ) target geometry; design doc in Phase 0.2.
