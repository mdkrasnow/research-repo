# Arxiv Weekly Sweep — diff-EqM Scoop Watch

**Active**: 2026-05-19 → 2026-10-01 (workshop + ICLR deadlines).
**Cadence**: Monday weekly.
**Owner**: claude (autonomous).

## Standing keywords

- `adversarial energy-based`
- `adversarial training diffusion`
- `adversarial flow matching`
- `PGD score matching`
- `hard example mining generative`
- `equilibrium matching`
- `regression EBM`
- `adaptive hard negative mining` (VeCoR future-work direction; high-priority watch)
- `adversarial post-training`

## Watch authors (high scoop risk)

- **Lin lab (ByteDance Seed)**: Shanchuan Lin, Ceyuan Yang, Zhijie Lin, Hao Chen, Haoqi Fan. 3 papers in 12 months (AFM/CAFM/AAPT). Likely to publish follow-up before Oct 1.
- **Du group**: Yilun Du (EqM senior author). Possibly affiliated with raywang4 (EqM first author). Watch for adversarial-EqM follow-ups.
- **Hong et al. (JIIOV Technology)**: VeCoR. Could publish v2 with "adaptive hard-negative mining" — their own future-work direction = exact v10 idea.

## Decision rules

- **HIGH threat hit** (covers v10 mechanism + regression flow / EqM): upgrade to per-paper note. Update SYNTHESIS.md §4. Escalate to `pipeline.json:needs_user_input` if differentiation collapses.
- **MEDIUM hit** (adjacent but distinct): log here only. Add to related-work citation list.
- **LOW / INFO**: log title + arxiv ID; no action.

---

## Week of 2026-05-19 (Phase 0)

### Searches run
- "adversarial flow matching 2025 2026" → found Lin AFM (2511.22475), CAFM (2604.11521), AAPT (2506.09350), FAIL (2602.12155), FMVP (2601.02228), Adversarial Flow Models v2.
- "PGD adversarial training combined GAN discriminator generative" → Rob-GAN (1807.10454), older defense work.
- "adversarial training consistency model single step" → ACT (2311.14097), AAPT.
- "Lipschitz regularization diffusion model" → singularities + score-regularity work.
- "hard negative mining diffusion generative 2025 adversarial" → no direct flow/EqM hits.
- "equilibrium matching follow-up 2026" → FlowEqProp (2604.08150, unrelated to EqM-Wang), under-review ICLR-2026 (2506.01158, unknown).
- "adversarial energy-based model diffusion 2025 2026" → DAT (2510.13872), AEBM-Diff (2403.01666).

### Top finds (already deep-read in Phase 0)
- HIGH: Lin AFM (2511.22475) — ICML 2026. Differentiation locked in SYNTHESIS.md.
- HIGH: Lin CAFM (2604.11521) — Apr 2026. Differentiation locked.
- HIGH: Wu DAT (2510.13872) — Oct 2025. Differentiation locked.
- MED: Kong ACT (2311.14097) — CVPR 2024. Confirmed distinct (no PGD).
- MED: VeCoR (2511.18942) — explicit future-work cite for our direction.
- INFO: AEBM-Diff (2403.01666) — gen-disc adversarial EBM, distinct.

### To investigate next sweep
- arxiv:2506.01158 — under ICLR 2026 review. Title unknown from search. Check abstract.
- Adversarial Flow Models v2 — verify if substantive update or formatting only.
- VeCoR v2 if posted (watch for new arxiv version).

### Threat assessment
**No new scoop-level findings beyond what is already accounted for in SYNTHESIS.md.** Branch B-Both remains viable. v10's specific mechanism (PGD-on-EqM-regression-target × CAFM-discriminator composition) is unclaimed.

---

## Week of 2026-05-26 (Phase 0 wrap-up + Phase 1a launch)

TBD — fill at next sweep.
