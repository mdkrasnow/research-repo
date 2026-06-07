# v17→EqM bridge — smoke postmortem (2026-06-07): mechanism FAILS on real CIFAR, NOT submitted

**Outcome:** built `v14_multi_morphism_aug` (the v17 recipe ported to CIFAR EqM), ran the mandatory
discovery + anchor-separability smoke per the diff-EqM smoke rule, and the mechanism FAILED the smoke. Per
discipline (mechanism-check before compute; max 1 retune; don't burn GPU on a failed smoke) **no FID job was
submitted**. The bridge code is preserved for record.

## What was built
- `projects/diff-EqM/experiments/dganm_variants/_multi_morphism.py` — v17 morphisms, PCA-whitened random-conv
  anchor, EMA-bandit policy + discover, ported to CIFAR [-1,1].
- `dganm_variants/v14_multi_morphism_aug.py` — Stage-1 discover frozen multi-family policy vs anchor
  (visible = center-zoom CIFAR, anchor = full CIFAR → bridge gap), Stage-2 EqM aug.
- configs `bridge150_v14_{discovered,random}.json`, `bridge_v14_smoke.json`.

## The failure (smoke evidence)
The load-bearing v17 property — a LABEL-FREE anchor that separates valid (on-manifold) morphisms from
invalid decoys at AUC 1.0 — **does NOT hold on natural CIFAR textures.** Per-family separability (valid
should score below all decoys):

| anchor | catches | MISSES (decoy accepted / valid rejected) | sep |
|---|---|---|---|
| random-conv ED | crop_erase, color_collapse | **big_shear accepted** (ED 0.10, lowest of all) | −0.57 |
| in-domain AE recon | big_shear | **crop_erase + color_collapse accepted** (easy to reconstruct) | −0.03 |
| combined max(z_conv, z_AE) | all 3 decoys | **valid `hue` rejected** (AE penalizes recolored images) | −0.42 |

Each anchor leaks a *different* family; the two base anchors are complementary on decoys but no single (or
naive combined) label-free anchor cleanly separates valid-from-decoy without per-CIFAR hand-tuning (e.g.
dropping `hue`). Discovery on the leaking anchor concentrated on a decoy (`big_shear` effective-usage 0.95,
decoy_usage 0.96) — submitting would have trained EqM on object-destroying shear, the exact CAFM-341-style
disaster the smoke rule exists to catch.

## Why (root cause)
v17's gym / MNIST / dSprites are CLEAN low-complexity manifolds where a destructive transform is obviously
off-manifold to a cheap feature anchor. Natural CIFAR textures are rich enough that a strong shear or a
center-crop still looks plausible to a random-conv or a small AE — the anchor is "not symmetry-aligned" on
natural images (documented earlier; see [[diff_eqm_symmetry_ladder]] feature-proxy note). This is the v17
write-up's explicitly stated risk realized: real image symmetries may not be a low-dim group separable in a
cheap label-free feature space.

## Decision & recommendation
- **Do NOT run the CIFAR bridge as-is.** Mechanism not ready; smoke correctly blocked it.
- Two honest forward paths (user's call):
  1. **Validated-safe lever:** known-symmetry augmentation (translate/crop) for the EqM aug — the toys + v17
     already say this is the safe CIFAR lever. Pairs the v17 *positive* (write-up) with a known-aug bridge.
  2. **Research fix (proxy-gated, NOT GPU yet):** a better label-free validity model that separates valid
     from ALL decoys on natural images — candidates: combined anchor + drop/penalize unreliable families,
     or a one-class density model. Gate it on the cheap per-family separability test (sep > 0) BEFORE any
     EqM run. (A Track-B *semantic* encoder would separate easily but violates the no-pretrained-semantics
     constraint.)
- The v17 Phase 0–3 positive STANDS on its own (synthetic + MNIST + dSprites + EqM-lite). The CIFAR bridge
  is a separate, harder step that this smoke shows is not yet mechanism-ready.

## Status
No active_runs added (nothing submitted). Bridge code committed for record + reproducibility of this
postmortem. pipeline.json updated: bridge BLOCKED on mechanism (anchor transfer), not on cluster/approval.

---
## RESOLUTION (2026-06-07): detector cracked — bridge now passes the decoy-avoidance smoke

Per user direction ("take a crack at the better detector first"), built a natural-image-robust label-free
validity detector (`experiments/v17_cifar_detector.py`):
- **robust AE**: a small autoencoder trained on CIFAR + MILD valid nuisances (flip / small affine /
  brightness / hue). It reconstructs VALID morphisms well (low error) and destructive decoys poorly. This
  closed the `hue` false-reject that broke the earlier plain-AE combination.
- **combined detector** = `conv-ED` (catches crop_erase + color_collapse) + robust-AE recon (catches
  big_shear). Static per-family separability: **sep = +0.378** — valid families all below all decoys
  (D1 plain-AE −0.026, D2 robust-AE alone −0.018, **D3 combined +0.378**).

Wired into discovery (`_multi_morphism.discover(ae=..., ae_weight=...)`): the combined penalty is a SUM, so
the AE veto must outweigh big_shear's low conv-ED — needs `ae_weight ≈ 50`:
- ae_weight 8 → decoy_usage 0.92 (big_shear still wins)
- ae_weight 25 → 0.157
- **ae_weight 50 → decoy_usage 0.079 (< 0.10 gate); no decoy in top families** (favors valid hue/bright/
  translate).

**Status:** the CIFAR bridge now PASSES its mechanism/decoy-avoidance smoke. `v14_multi_morphism_aug`
defaults updated (use_ae=true, ae_weight=50). REMAINING gate is the human FID-approval to submit the
150ep×4-arm CIFAR bridge (FID never auto-authorized). The mandatory ≥16-sample probe still runs as the
short `bridge_v14_smoke` gpu_test job before the full run.
