# SiT: Scalable Interpolant Transformers

**Citation**: Ma, Goldstein, Albergo, Boffi, Vanden-Eijnden, Xie. *SiT*. ECCV 2024. arxiv 2401.08740.

## Method
Generalizes DiT via interpolant framework. Flexible distribution-connecting paths. FID-50K 2.06 IN-256 best.

## Relevance
- **EqM uses SiT backbone exactly** (Wang 2025 §method). Removing time conditioning = EqM.
- **Lin CAFM uses SiT** for primary results.
- Code: github.com/willisma/SiT.

## Action
- Phase 5 head-to-head uses SiT directly.
- Cite as architecture lineage.
