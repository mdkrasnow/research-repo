# Literature Read Slate — Phase 0

**Locked**: 2026-05-19. Branch B-Both context.
**Total**: 20 papers. Aim ~3 work-days of reads across parallel WebFetch.

Legend:
- **F** = full-read (HTML body + tables)
- **A** = abstract + method
- **S** = skim
- **Threat**: HIGH (scoop / direct competitor) | MED (overlapping ideas) | LOW (background) | INFO (cite for completeness)

## Bucket 1 — Direct competitors / scoop check (HIGH)

| # | Paper | Arxiv | Depth | Threat | Status |
|---|---|---|---|---|---|
| 1 | Lin et al. — Adversarial Flow Models (ICML 2026) | 2511.22475 | F | HIGH | METHOD READ — extend |
| 2 | Lin et al. — Continuous Adversarial Flow Models | 2604.11521 | F | HIGH | METHOD READ — extend |
| 3 | Wu et al. — Scalable EBMs via Adversarial Training (DAT) | 2510.13872 | F | HIGH | partial — need full table |
| 4 | Geng et al. — Improving Adversarial EBM via Diffusion | 2403.01666 | F | HIGH | abstract only — need method |
| 5 | Wang et al. — Equilibrium Matching (EqM base model) | 2510.02300 | F | INFO | summary only — need method |
| 6 | Kong et al. — ACT: Adversarial Consistency Models (CVPR 2024) | 2311.14097 | F | HIGH | unread |

## Bucket 2 — Adversarial training for generative models (MED)

| # | Paper | Arxiv | Depth | Threat | Status |
|---|---|---|---|---|---|
| 7 | Madry et al. — Towards Deep Learning Resistant to Adversarial Attacks (PGD AT) | 1706.06083 | A | INFO | unread |
| 8 | "What is Adversarial Training for Diffusion Models?" | 2505.21742 | F | MED | unread — equivariance argument relevant to v01-vs-v10 |
| 9 | Jiang et al. — VeCoR (velocity contrastive reg) | 2511.18942 | A | MED | unread — closest CIFAR prior to v02 |
| 10 | Luo et al. — Diffusion Contrastive Divergences (DCD) | 2307.01668 | A | MED | unread — mining at noised inputs |
| 11 | Liu et al. — Rob-GAN (PGD+GAN combination) | 1807.10454 | A | MED | unread — closest "combine" prior, defensive |

## Bucket 3 — EBM + flow/regression connections (MED/INFO)

| # | Paper | Arxiv | Depth | Threat | Status |
|---|---|---|---|---|---|
| 12 | Du & Mordatch — Implicit generation & modeling with EBMs | 1903.08689 | A | INFO | unread |
| 13 | Grathwohl et al. — JEM (classifier as EBM) | 1912.03263 | A | INFO | unread |
| 14 | Lipman et al. — Flow Matching | 2210.02747 | F | INFO | unread — foundational for SiT |
| 15 | Song et al. — Consistency Models (ICML 2023) | 2303.01469 | A | INFO | summary only |

## Bucket 4 — SiT / DiT background (needed for Phase 5)

| # | Paper | Arxiv | Depth | Threat | Status |
|---|---|---|---|---|---|
| 16 | Ma et al. — SiT (Scalable Interpolant Transformers) | 2401.08740 | F | INFO | unread |
| 17 | Peebles & Xie — DiT | 2212.09748 | A | INFO | unread |

## Bucket 5 — Hard-example mining priors (LOW)

| # | Paper | Arxiv | Depth | Threat | Status |
|---|---|---|---|---|---|
| 18 | Shrivastava et al. — OHEM | 1604.03540 | S | INFO | unread |
| 19 | Lin et al. — Focal Loss | 1708.02002 | S | INFO | unread |

## Bucket 6 — Watch list (Lin-lab follow-ups + adjacent)

| # | Paper | Arxiv | Notes |
|---|---|---|---|
| W1 | AAPT (autoregressive adversarial post-training, video) | 2506.09350 | Lin-adjacent — video AT |
| W2 | FlowEqProp | 2604.08150 | confirmed unrelated to EqM-Wang |
| W3 | FAIL (flow matching adversarial imitation) | 2602.12155 | likely LOW relevance, verify |
| W4 | FMVP (adversarial purification with FM) | 2601.02228 | defensive, LOW |

## Execution order

**Round 1 (parallel WebFetch, ~6 papers at a time):**
1. Wang EqM full method (#5) — load-bearing for v10 mechanism design
2. Kong ACT full (#6) — closest adversarial-discriminator-+-regression prior; high differentiation impact
3. "What is AT for Diffusion?" (#8) — equivariance argument may invalidate v10 hypothesis
4. Wu DAT full (#3) — finish full method extract
5. Geng AEBM-Diff full (#4)
6. VeCoR (#9) — closest v02 mechanism prior

**Round 2:**
7. Lipman Flow Matching (#14) — foundational for SiT understanding
8. Consistency Models full method (#15)
9. SiT (#16) — Phase 5 head-to-head dependency
10. Luo DCD (#10) — noised-input mining prior
11. Madry PGD (#7) — foundational
12. Rob-GAN (#11) — PGD+GAN combination prior

**Round 3 (skim):**
13–19. Du, JEM, DiT, OHEM, Focal, watch-list entries.

**Synthesis (after Rounds 1+2)**: write SYNTHESIS.md.

## Pre-assessment risks to validate during reading

1. Does "What is AT for Diffusion?" argue equivariance ≠ invariance in a way that **invalidates v10's L2-on-target objective**? (v01 hinge was invariance-leaning; v10 may be neither — verify.)
2. Does Kong ACT already do "discriminator + regression target" combination on consistency models (≈ Branch B but different model family)? If yes, weakens our combination novelty.
3. Does VeCoR test PGD-on-input variants in ablations? If yes, narrows v10 novelty.
4. Does Luo DCD's "mining at noised inputs" overlap with v10 mechanically? Need to check loss form.
5. Are there any 2026 Lin-lab follow-ups already on arxiv we missed?
