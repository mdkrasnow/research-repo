# Hard-Positive Symmetry Mining (HPSM) — Theory + Pre-Registered Ladder

**Date:** 2026-06-11

## Theory: the positive dual of v10 ANM

| | v10 ANM (existing) | HPSM (this) |
|---|---|---|
| mines | hard INVALID/noised examples (δ in noise space) | hard VALID named symmetries (T in image space) |
| trains field to | REJECT them (robustness off-trajectory) | ACCEPT/EQUALIZE them (equivariance) |
| adversary space | noised perturbation | data symmetry (firewall-constrained) |

    T* = argmax_T [ hardness(T) + gap_reward(T) ]   s.t. T valid (decoy/move/mag firewall)
    L  = L_base(x) + lam_hpsm·L_eqm(T*(x)) + lam_consistency·commutator(x, T*(x))

**Why `gap_reward` is load-bearing.** Loss-hardness ALONE ties random on full CIFAR (the ASM CPU verdict,
2026-06-09: no gap → any valid op is equally hard). `gap_reward = ED(x) − ED(T(x))` rewards moving toward
the anchor manifold — the targeting signal that lets HPSM lock the missing factor (e.g. saturate on a
desaturated→full gap, the gap15 win). On full CIFAR (no gap) gap_reward ≈ 0, so HPSM is expected to ≈
random there too — the honest prior. Named library only (operator expressiveness is NOT the bottleneck —
targeting + validity is).

**Consistency = commutator.** `commutator(x,T) = ||F(T(x)_t) − J_T F(x_t)||²`, minimized so the field
becomes equivariant to the mined valid symmetry. The novel EqM-native signal.

## Code
- `dganm_variants/hpsm_miner.py` — miner (score = hardness + gap_reward − invalid − decoy − collapse + entropy).
- `dganm_variants/v18_hpsm_morph.py` — EqM variant (lam_hpsm=0 ⇒ base).
- `dganm_variants/v19_anm_hpsm_hybrid.py` — v10 ANM + HPSM (lam_anm=0 ⇒ HPSM; reuses v10 PGA).
- `symmetry-discovery/experiments/hpsm_ladder.py` — gated CPU ladder A/B/C.
- configs: `hpsm_smoke / hpsm_color_gap_cpu / hpsm_full_cifar_150 / hybrid_full_cifar_150` (.json).

## Gated ladder (pre-registered — no GPU until CPU passes; no threshold tuning after seeing FID)

| stage | gate | on pass | on fail |
|---|---|---|---|
| A unit/validity | transforms shape/range; miner valid T*; decoy_usage<0.05 (high penalty); v18+v19 fwd/bwd no-NaN, components nonzero; lam=0 reductions | ADVANCE | REPAIR miner/firewall |
| B color-gap (desat→full) | HPSM or static selects `saturate`; decoy<0.05; HPSM eqm_full beats random by ≥0.005 abs or 1% rel | ADVANCE | REPAIR scorer/gap_reward (sat not sel) / firewall (decoys) / STOP (no payoff) |
| C full-CIFAR TinyEqM | SOLO: HPSM<random & base; HYBRID: v10+HPSM<v10 | PROMOTE_TO_GPU | STOP — gap15 flagship, go to Stage E 3-seed |
| D GPU full ladder | v00/v10/random-valid/HPSM/hybrid, v10 protocol; ep50/100/final FID | per success hierarchy below | — |
| E Track-A 3-seed | gap15 seeds 1,2: discovered beats crop ≥2/3 seeds, beats random 3/3 or clear mean margin | flagship significance locked | — |
| F | scoped learned exp(A) family ONLY if D/E show named library is the bottleneck | — | — |

**Success hierarchy (Stage D):** min = HPSM<random; strong = HPSM<base & approaches v10; major = HPSM<v10;
best-paper = hybrid<v10.

**Prior (honest):** full CIFAR has no gap → expect HPSM ≈ random there (Stage C likely STOP), gap15 stays
the flagship; the real HPSM value is gap-conditional. Stage D launches only if C passes OR the queued
Track-B ladder needs completion.

## Results log
- **Stage A: PASS → ADVANCE** (2026-06-11). 13 families ok; T*=hue valid; decoy_usage 0.0 (high penalty,
  decoys 100% rejected); v18 base+hpsm+cons all finite, grad>0, lam0⇒base; v19 base+anm+hpsm all nonzero,
  grad>0, anm0⇒hpsm.
- **Stage B: payoff gate FAIL → STOP (record negative)** (2026-06-11). desat→full color gap, TinyEqM eqm_full:
  static_gapaware 0.4114 (saturate) < HPSM_loss 0.4174 (saturate) < random 0.4189 < base 0.4227;
  HPSM_comm/loss_comm 0.4257 (picked hue, worse).
  - TARGETING WORKS: HPSM_loss + static both select `saturate`, decoys avoided (firewall fine, gap_reward fine).
  - PAYOFF FAILS: HPSM_loss beats random by only 0.0016 (< 0.005 abs / 0.0042 rel gate). Per spec "selected
    but no payoff → STOP, record negative." NOT a repair (nothing broken — targeting + firewall both work).
  - KEY: **static gap-aware discovery CLEARS the payoff bar (+0.0075 over random); adversarial HPSM does
    NOT.** Making the objective adversarial adds nothing over the simpler static gap-aware approach (same
    lesson as the ASM full-CIFAR verdict). Commutator mode again mis-targets (hue not saturate) — the
    field-commutation signal does not isolate the missing factor on a lightly-trained probe.
  - DECISION: STOP. No Stage C/D GPU for HPSM-solo. Static gap-aware (gap15) stays the flagship; HPSM's
    adversarial framing is not justified over it on CPU evidence. No silent HP tuning.
- Stage C: SKIPPED (Stage B STOP — full CIFAR already known to tie random per ASM verdict; no new GPU).
