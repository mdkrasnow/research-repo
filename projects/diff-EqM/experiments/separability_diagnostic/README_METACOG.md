# Metacognition method-improvement suite (EqM B/2, inference-time)

Goal: turn the *diagnostic* result (early descent-shape predicts EqM sample
quality) into *method* improvements — better ways to spend a fixed sampling
compute budget, with NO weight changes and NO test-time image access.

## Conservative assumptions (built AFK; revisit if wrong)
- Matched-compute target = **NFE/img = 750** (R=3 draws × 250 GD steps), the
  locked Pareto/CI setting. Promotion baseline = **probe_k50** (the locked early
  selector, FID 24.66±0.16 @50k in the 5-seed CI).
- "Matched NFE" counted with the pareto convention (a draw = `steps` NFE; the GD
  loop does `steps-1` forwards). Selection arms are EXACT 750 by construction;
  segmented arms report MEASURED forwards (≈R·(steps-1)=747 budget-neutral, more
  if they add corrector evals) and the aggregator flags any |nfe-750|/750>2%.
- Screen at seed0, n=10k (cheaper than 50k, enough to rank arms). Promotion →
  full 50k×5-seed paired vs probe_k50/energy_path/random/long250.
- Stacked ranker + probe are trained on the cached good/garbage **DEV** pool
  (same labels the probe used). FROZEN at test. No test-set oracle anywhere.

## Two engines (run_metacog_policy_sweep.py)
- **selection** — run R full draws + read partial-traj features; keep argmin
  `policy.score`. NFE/img = R·steps EXACT; identical seeding to `pareto_sample.py`
  so `energy_path` reproduces the pareto seed0 (repro check in the aggregator).
- **segmented** — run lanes in segments; at read steps apply per-lane actions
  (continue/restart/churn/eta/heun). Engine counts every model() forward →
  exact measured NFE.

## Policies (metacog_policies.py)
Selection (matched-NFE screen, launched tonight):
1. `stacked_selector` — calibrated logistic over [probe@k, log Σ‖f‖, log‖f‖_end,
   norm slope, oscillation, decay]; keep min risk. (frozen, DEV-fit)
2. `smc_metacog` — risk-weighted stochastic selection over R particles
   (softmax(−β·risk)). With keep-1 + matched NFE, SMC reduces to risk-weighted
   resampling among the R draws — extra particles/jitter are NOT affordable at
   matched compute, so the full-jitter SMC is an extra-NFE variant (future).
8. `multiread_triage` — read 50/75/100; demote lanes flagged bad at k=50, keep
   best by risk@100; fall back to argmin@100 if none survive.
Baselines in-screen: `random` (null), `energy_path` (best trivial), `probe_k50`
(locked selector = promotion baseline), `vanilla` (keep draw 0).

Segmented (smoked tonight at n=256; promote to full screen only if NFE/no-crash clean):
4. `risk_compute_allocator` — restart the worst risk-fraction of lanes.
5. `optimizer_switch` — failure-type→action (oscillatory→η↓, high-mag→Heun, else restart).
6. `churn_rescue` — high-risk lanes get a noise kick, then continue.
7. `heun_corrector` — high-risk lanes use a 2-eval Heun step (off-budget by design).
9. `branch_lookahead`, 3. `cem_elite`, 10. `oracle_imitation_dev_only` — specced
   in the policy module's design notes; need extra-NFE candidate generation, so
   they run as labeled extra-NFE tiers (not in the matched screen). Not launched
   tonight (would need the segmented engine's extra-draw path hardened first).

## Literature basis
- **EqM** (optimization-based sampling / adaptive compute): treats sampling as
  gradient descent on a learned field → restart/early-abandon/step-size control
  are all legitimate inference-time levers.
- **SMC** (sequential Monte Carlo): particle weighting + resampling by a risk
  signal; `smc_metacog`.
- **CEM** (cross-entropy method): fit an elite distribution, resample; `cem_elite`.
- **Langevin / EDM** (stochastic correction / churn): noise injection to escape
  bad basins; `churn_rescue`.
- **DDIM / DPM-Solver** (sampler/corrector design): higher-order corrector steps;
  `heun_corrector`, `optimizer_switch`.

## Files
- `metacog_policies.py` — pure-numpy policies + features + stacked trainer.
- `run_metacog_policy_sweep.py` — GPU engine (selection + segmented), incremental FID.
- `aggregate_policy_sweep.py` — shards→FID, NFE table, promotion verdict, repro check.
- `test_metacog_local.py` — CPU fake-model smoke (NFE/paired-seeds/no-oracle). PASSES.
- `../../../scripts/cluster/deploy_metacog.sh` — builds stacked artifacts + submits.

## Reproduce
```
bash scripts/cluster/deploy_metacog.sh                      # via scripts/cluster/ssh.sh, repo root
# when done:
python projects/diff-EqM/experiments/separability_diagnostic/aggregate_policy_sweep.py \
  --root .../runs/b2_vanilla/metacog --baseline probe_k50 \
  --ref-stats projects/diff-EqM/results/in1k_reference_stats.npz \
  --pareto-energy .../runs/b2_vanilla/pareto_r3energy
```
Promotion: matched NFE (±2%) AND FID ≤ probe_k50 − 0.5. Promoted → 50k×5-seed paired.
```
