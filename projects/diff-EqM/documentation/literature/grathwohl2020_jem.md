# JEM: Your Classifier is Secretly an Energy-Based Model

**Citation**: Grathwohl et al. *JEM*. ICLR 2020. arxiv 1912.03263.

## Method
Reinterprets standard classifier as EBM: `p(x,y) ∝ exp(f(x)[y])`, `p(x) ∝ Σ_y exp(f(x)[y])`. SGLD sampling for generation.

## Relevance
- **Direct precursor to DAT (Wu 2025)**. DAT replaces SGLD with PGD-generated negatives.
- Background for "classifier-as-EBM" framing — orthogonal to our regression-based implicit-energy approach.

## Action
Cite as background for DAT lineage. Distinguish: JEM/DAT = classifier-as-EBM; v10 = implicit-energy gradient field (EqM).
