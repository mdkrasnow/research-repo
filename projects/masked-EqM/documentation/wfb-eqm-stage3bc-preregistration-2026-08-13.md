# Stage 3B / 3C pre-registration (written BEFORE Stage 3A results are known)

**Date**: 2026-08-13
**Written at**: commit `27c227a`, with Stage 3A (job 38985448) queued and no rows
returned yet. Pre-registered deliberately, so the intervention is chosen by the
classification rather than rationalized after seeing it.

Context: `wfb-eqm-stage3-audit-2026-08-13.md` (three-way audit + retraction).

## The decision this document binds

Stage 3A classifies the CG-FBGN failure as H1, H2, or H3. **Exactly one**
Stage 3B intervention follows, and it is fixed in advance per branch. No
branch may be rescued by tuning if it fails its own test; the maximum is
**one retune per failing direction** (CLAUDE.md hard rule).

| Stage 3A reads | classification | Stage 3B intervention | what would falsify the repair |
|---|---|---|---|
| median `R_B` < 0.5 or median `D_B` > 0.25 at `eta*`, **and** `d_V < 0` on most batches | **H1** local-model failure | damping sweep `lambda in {lambda_0, 10, 100}` (already folded into the 3A job) | stronger damping fails to move `D_B` down / `R_B` toward 1, or kills independent-batch descent |
| `R_B ~ 1` and `D_B << 1`, **but** `d_V >= 0` frequently, or trust loss rises even at the smallest `eta` | **H2** stochastic over-solving | model batch 8 → 32 as 4 deterministic microbatches, ONE stacked solve | batch 32 does not improve `d_V` sign rate, trust alignment `C_V`, or probe transfer |
| both fire | **H3** | smallest model batch that repairs `C_V`, PLUS adaptive LM damping | either component fails its own test above |

## Why the H2 branch needs a stacked operator, not an average

`M^T v = sum_j M_j^T v_j` couples the microbatches through the shared `theta`,
so the stacked `A = M M^T` is **not block diagonal**. Averaging J separate
solves computes `sum_j M_j^T (A_j + lambda I)^{-1} r_j`, which inverts each
block in isolation and discards exactly that coupling — a different algorithm,
and one that would have to be labelled as such.

The correct operator is implemented as `stacked_mixed_gram_mv` /
`compute_fbgn_gradient_cg_microbatched` in `fb_direct/exact_hvp.py`, validated
in `tests/test_stacked_microbatch_gn.py` against the explicit stacked Jacobian
`M M^T`, against `mixed_gram_mv` at `J = 1`, and — decisively — against the
average-of-solves shortcut, which it must *not* match. Cost is ~J× per CG
iteration; that is the honest price of the larger stochastic model.

Escalation rule: test `B_model = 32` first. Go to 64 **only** if 32 materially
improves the result but is clearly not enough.

## Damping, if H1 or H3 fires

Carry `lambda_t` **forward between steps** as an absolute quantity. Do not
recompute it each step as a fixed ratio of the current `lambda_max` — the whole
point of LM adaptation is that damping accumulates evidence about local model
trustworthiness, and (per the audit's §4) `lambda = rho * lambda_max` makes
`A(A+lambda I)^{-1}` invariant to uniform curvature rescaling, so a ratio rule
cannot respond to curvature at all.

Initialize `lambda_0 = 1e-4 * lambda_max(A)` at the transition checkpoint.
Update on the reduction ratio `R` (Martens 2010): strong agreement → shrink,
poor agreement → grow. The 1/4 and 3/4 thresholds are the starting point, not
constants to preserve if this normalization makes them inappropriate.

## STR-FBGN, built only if Stage 3B supports it

Per accepted step:

1. draw model batch `B_t` (size set by the H2 branch), build `p` from ONE
   stacked solve of `(A + lambda_t I) u = r`;
2. compute predicted source reduction from `r + M p`, measure the actual one,
   form `R_t`;
3. **local trust decision** — poor `R_t` → shrink/reject, increase `lambda_t`,
   with a *bounded* small number of retries on the same frozen model batch (not
   six unconstrained expensive re-solves);
4. **stochastic transfer decision** — draw a FRESH acceptance batch `V_t`
   (32–64 examples, corruption frozen for the paired before/after), separate
   from both the model batch and the global probe. Reject only on clear
   evidence of harm: report mean paired `dL`, its standard error, and the
   fraction of per-example improvements. Good `R_t` with a harmful `V_t` is a
   *different* failure from bad linearization and must not be treated as "raise
   lambda"; prefer rejecting the direction, enlarging the model batch on retry,
   or redrawing — which of these is set by Stage 3B.

The deterministic global probe `P` is **evaluation only** — never used for
acceptance, damping, step-size selection, or retries, at any point.

CG tolerance: start at a true relative residual of **0.05**, not 0.01–0.02. The
300-step run spent heavy compute solving a noisy 8-image system to 1e-3, which
is very likely counterproductive now that correctness is established. Recompute
the true residual periodically from a fresh operator application. Keep the
tighter tolerance only if Stage 3A shows 0.05 materially changes the direction.
No Krylov reuse, low-rank approximations, or stale curvature until STR-FBGN is
shown to train at all.

## Stage 3C — 50-step causal test (pre-registered gate)

Arms, all from the identical checkpoint, matched data order and corruption
seeds, no Adam: **F0** current CG-FBGN, **F1** repaired STR-FBGN, **D**
exact-direct control if cheap. 50 accepted steps or a capped number of
proposals. Log every proposal (source before/after, predicted and actual
reduction, `R_t`, `D_t`, model-batch size, acceptance-batch before/after,
accept/reject + reason ∈ {model-fidelity, stochastic-transfer, numerical},
`lambda_t`, `lambda_max` as diagnostic only, CG iters, true CG residual, `||p||`,
`||Mp||/||r||`), and the deterministic probe every 5–10 accepted updates.

**PASS requires all of:**

1. no strong upward probe trajectory (the 300-step run's signature);
2. preferably `dL_P < 0`;
3. same-batch progress remains real;
4. no numerical correctness failures;
5. rejection statistics make conceptual sense;
6. damping does not explode merely because raw `lambda_max` moves;
7. **the improvement is attributable to the diagnosed mechanism, not to freezing
   the model.** Explicitly compare `sum ||d theta||` and `sum ||d s_P||` between
   F0 and F1. A method that "stabilizes" by making every step vanishingly small
   is a **FAIL**, not a pass.

Only a clean pass authorizes a matched 300-step run. Ambiguous → stop and report.

## Falsification / kill condition

Kill the FBGN thread if, with moderately larger model batches **and** sensible
damping **and** accurate solves, the FBGN direction is still frequently an
independent-batch ascent direction or still consistently worsens the
deterministic probe. Do not keep adding machinery. The negative result is
itself publishable-grade evidence:

> Mixed-Jacobian whitening explains and suppresses the gradient-spike mechanism,
> but Gauss-Newton optimization of the scalar EqM field does not provide a
> useful stochastic training trajectory at this checkpoint.

Note the audit's finding that the raw-direct negative control ALSO worsens the
probe (+0.208) strengthens this branch: if the failure is a property of
single-minibatch second-order-ish optimization at this checkpoint generally,
that is the honest result, and no FBGN-specific repair will recover it.

## Not the next thread

Backbone/structural damping stays a **fallback**, not the current experiment.
It is only revisited if (1) larger model batches do not fix transfer, (2) LM
damping still leaves large `D_B`, and (3) model-agreement failure clearly
correlates with large internal representation changes. No further QKV/head
localization work now.
