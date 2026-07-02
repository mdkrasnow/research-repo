# Metacognition sampler — math formulation + energy-OOD proposal (for Yilun, draft 2026-07-02)

## 1. Setup

EqM trains a field `f_θ(x_γ)` on interpolant `x_γ = γ x1 + (1-γ) x0`, `x0 ~ N(0,I)`, target

```
target(x1, x0, γ) = (x0 - x1) · c(γ)
c(γ) = min( 1 - (1-1/interp)γ,  1/(1-interp) - γ/(1-interp) ) · 4,   interp = 0.8
```

(transport.py:122-126). `c(γ) → 0` as `γ → 1` (data manifold), so `f` is a **regression
target field**, not a normalized score — no explicit energy potential in the vanilla
parameterization (energy-EqM variant exists but untested here).

Sampling = GD descent on `f`: `x_{k+1} = x_k - η f_θ(x_k)`, `k=0..K`, `η` fixed step, `K=250`
typical. This produces, per sample `i`, four scalar trajectories over `k`:

- `norm_i(k) = ‖f_θ(x_k)‖`
- `dot_i(k) = ⟨f_θ(x_k), x_k⟩` (or normalized cosine variant)
- `l2_i(k) = ‖x_k‖`
- `step_dot_i(k) = ⟨f_θ(x_k), x_{k-1}-x_k⟩` (per-step alignment)

## 2. What was tried and killed: energy as a quality score

Two energy-style scalars, both **dead** (within-norm-bin AUROC, de-confounded vs `‖f‖`):

- **Dot energy**: `s1 = -dot_i(K)` (terminal alignment) → AUROC 0.609
- **Path-integral energy**: `s3 = -∫_0^K dot_i(k) dk` (Lyapunov-style accumulated alignment)
  → AUROC 0.605

Both **below** a dumb non-model baseline (`s4`, k-NN distance in raw pixel/latent space to
real-data manifold, no `f` access at all) = 0.627. Conclusion already logged: **`f`'s implicit
energy carries less good/bad information than a trivial geometric distance**. This is
consistent with EqM's target having `c(γ)→0` at the manifold — near convergence the *magnitude*
of `f` is forced toward zero for good and bad samples alike, so a scalar read at/near the
fixed point is uninformative by construction. Any single-timestep energy read inherits this
degeneracy.

## 3. What works: a linear probe over trajectory SHAPE (not magnitude, not any single scalar)

Define, per sample, the **normalized** (magnitude-removed) curves:

```
n̂(k) = norm(k) / norm(0)          (down-sampled to 16 bins)
d̂(k) = dot(k)  / |dot(0)|         (down-sampled to 8 bins)
```

plus 6 hand dynamics features computed from the raw `norm`, `dot` curves:

```
osc_n   = frac_k[ sign(Δnorm(k)) ≠ sign(Δnorm(k-1)) ]     (oscillation rate, norm)
osc_d   = same, on dot
slope   = OLS slope of log(norm(k)) vs k                   (log-decay rate)
late    = mean(n̂ over last T/4) - mean(n̂ over first T/4)   (net settle amount)
rel_jit = std(Δnorm) / mean(norm)                           (relative jitter)
curv    = mean_k |Δ²n̂(k)|                                    (2nd-difference curvature)

SHAPE feature vector: φ_shape(i) = [osc_n, osc_d, slope, late, rel_jit, curv,  n̂(16 bins), d̂(8 bins)]  ∈ R^30
MAG feature vector:   φ_mag(i)   = [norm(K), mean(norm), max(norm), min(norm), l2(K), dot(K), Σstep_dot]  ∈ R^7
```

A **linear (logistic) probe** `p(i) = σ(w·φ_shape(i) + b)` trained on
good-vs-garbage labels (Inception pool3 k-NN distance to 20K real images, thresholded at
tail quantiles) via 5-fold CV, evaluated **within grad-norm-quantile bins** to strip out any
magnitude confound:

```
AUROC_shape (de-confounded) = 0.813 ± 0.002  (5 seeds)     — held-out end-to-end: 0.818 ± 0.012
AUROC_mag   (magnitude-only floor)            ≈ chance-to-weak
AUROC_full  (mag + shape, upper bound)         ≈ AUROC_shape (mag adds ~nothing — confirmed by
                                                  method-sweep: "stacked" policy < probe_k50 alone)
```

**Formal claim**: bad/good separability is not encoded in any pointwise statistic of `f`
(energy, norm, dot at any single `k`) but in the *second-order temporal structure* of the
descent — how much the field oscillates / fails to monotonically settle. This is a
metacognitive signal about the *optimization process*, not about the *learned potential*.

## 4. The deployed selector

Best-of-R with early stopping ("probe@50"): run `R` independent GD chains per output, read
`φ_shape` at step `k=50` (partial trajectory — sufficient, ~= end-of-trajectory AUROC per
partial-probe test), score each with the trained `p(i)`, keep the chain with lowest predicted
garbage-probability, discard the rest, finish only the survivor to `K=250`. Equal-NFE variant:
restart the flagged chain from a fresh `x0` instead of running `R` in parallel.

Result (5-seed CI, IN-1K B/2, n=50K): `probe@50` FID `24.66±0.16` vs random-restart control
`27.95±?`, paired Δ +3.29 FID, t≈75 (all seeds), beats depth-matched vanilla-GD (+3.44) and
energy-path selector (+1.04). This is the number the paper leads with.

## 5. Yilun's ask #2: can the energy function itself be made to distinguish OOD/garbage?

Current status: **no** — vanilla EqM's dot/path energy is dead as a per-sample discriminator
(§2). Three concrete levers to make the *energy itself* informative, in order of how much they
touch the trained model vs. just the read-out:

**(a) Read-out fix — integrate over the right window, not the whole path or the endpoint.**
The path-integral (§2, s3) averages over all `k`, diluting a signal that (per §3) is
concentrated in *early-to-mid* dynamics (oscillation, settle-rate), and the terminal dot
(s1) reads exactly where `c(γ)→0` forces convergence for both classes. A windowed energy
`s_window = -∫_{k1}^{k2} dot(k) dk` restricted to the region where §3's `slope`/`osc` features
live (roughly first half of descent) is untested and cheap (re-derivable from existing
trajectory shards, no retrain) — direct next experiment.

**(b) Loss-side fix — train an explicit energy with a margin/contrastive term so trajectory
shape gets folded INTO a scalar at train time**, instead of asking a frozen vanilla field to
expose it post-hoc. Two compatible options per the EqM-compatibility rules (AGENTS.md):
  - **Low-risk**: auxiliary loss = base EqM loss evaluated on hard-mined near-OOD points
    (reuse v10 PGD-mining infra) *combined with* a small regularizer pushing `Var_k[dot(k)]`
    (the oscillation the probe already found predictive) to be discriminative between
    mined-hard and clean populations at train time — i.e., distill the probe's shape signal
    into the field's own dynamics rather than reading it out after the fact.
  - **Higher-risk, needs written compat argument before code**: an explicit energy-EBM head
    `E_θ(x)` trained with an NCE/InfoNCE contrastive objective against `c(γ)`-scaled
    negatives (real EqM-E variant, currently untested per memory `eqm_separability_diagnostic`
    caveat "explicit-energy EqM-E untested but is the paper's fragile mode" — flagged fragile,
    treat as v1 exploratory only, mechanism check required before any train job).

**(c) Distillation fix — train a second, small model whose *only* job is to predict the
probe's shape-derived label from a SINGLE forward pass** (e.g., predict `p(i)` from `x_γ` at
one fixed `γ`, or from `f_θ(x_γ)` alone), i.e. compress the temporal probe into a scalar
"energy-like" function by supervised distillation rather than hoping it falls out of `f`'s
geometry. This turns "does the energy function distinguish OOD" into a well-posed supervised
problem with a known-achievable target (0.813 AUROC ceiling from the temporal probe) instead
of an open discovery question. Cheapest of the three, reuses existing labels/trajectories,
recommended next step if PI wants "energy that works" specifically vs. "selector that works"
(current paper claim already covers the latter).

## Recommendation

Ship (a) first — it's a re-read of already-collected trajectory shards, ~zero GPU cost,
answers directly whether *energy specifically* (not the general shape probe) can be rescued
with a better window. If (a) doesn't cross the 0.80 bar, (c) is the next cheapest (distillation,
supervised, bounded by known ceiling) before touching the training loss per (b).
