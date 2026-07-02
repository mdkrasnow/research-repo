# Metacog method-improvement sweep — launch record (2026-06-28)

Built + launched AFK. Suite turns the diagnostic (early descent-shape predicts EqM
quality) into inference-time method gains: better ways to spend a fixed NFE budget,
no weight changes, no test-time pixels/oracle. Full design: `README_METACOG.md`.

## Status: LOCAL SMOKE PASSED, SCREEN LAUNCHED, awaiting GPU
- `test_metacog_local.py`: ALL CHECKS PASS — selection NFE/img == R·steps EXACT;
  segmented churn budget-neutral (177 = R·(steps−1)); heun off-budget (>177);
  paired-seed determinism; no test-time oracle/label tokens in run path.
- Stacked frozen DEV rankers built on cluster: `stacked_artifact_k{50,75,100}.npz`.

## Launched jobs (git f3f008e)
Selection screen — matched NFE=750, seed0, n=10k:
| arm | job | partition |
|---|---|---|
| random (null) | 25649517 | gpu |
| energy_path (best trivial) | 25649518 | seas_gpu |
| probe_k50 (promotion baseline) | 25649540 | gpu |
| stacked_selector | 25649541 | seas_gpu |
| smc_metacog | 25649542 | gpu |
| multiread_triage | 25649553 | seas_gpu |

Segmented real-model smokes — n=256 (NFE/no-crash validation before full screen):
| arm | job | partition |
|---|---|---|
| churn_rescue | 25649574 | gpu |
| heun_corrector | 25649575 | seas_gpu |
| optimizer_switch | 25649576 | gpu |
| risk_compute_allocator | 25649577 | seas_gpu |

All PENDING(Priority) at submit, clean (no init/import/nvidia-smi failures). Each
4-GPU, nvidia-smi guarded (`|| exit 99`), incremental-FID shards only.

## On completion (next session / human)
1. Reconcile `active_runs` vs `squeue`/`sacct`; move finished → `completed_runs`.
2. Aggregate:
```
python projects/diff-EqM/experiments/separability_diagnostic/aggregate_policy_sweep.py \
  --root /n/home03/mkrasnow/research-repo/projects/diff-EqM/experiments/separability_diagnostic/runs/b2_vanilla/metacog \
  --baseline probe_k50 \
  --ref-stats /n/home03/mkrasnow/research-repo/projects/diff-EqM/results/in1k_reference_stats.npz \
  --pareto-energy /n/home03/mkrasnow/research-repo/projects/diff-EqM/experiments/separability_diagnostic/runs/b2_vanilla/pareto_r3energy
```
   - REPRO check: sweep `energy_path` FID should match pareto seed0 energy_path
     (guards the incremental-FID pipeline). If MISMATCH → debug before trusting.
3. Promotion rule: matched NFE (±2%) AND FID ≤ probe_k50 − 0.5.
   - Promoted selection arms → full 50k×5-seed paired vs probe_k50/energy_path/
     random/long250 (extend `deploy_metacog.sh` to seeds + n=50000).
   - Segmented smokes clean (NFE as expected, samples sane) → promote those
     policies to the n=10k matched screen as a second wave.
4. If NOTHING beats probe_k50: that is itself the result — at matched compute the
   locked early-shape selector is the ceiling among these mechanisms; report the
   table and stop (no laundering). Update `SELECTOR_LOCKDOWN_RESULTS.md`.

## Overnight progress (2026-06-29)
- Selection arms passing CLEAN at matched NFE 750.0 exact: energy_path, stacked_selector,
  multiread_triage done (n=10k). random/probe_k50/smc_metacog stuck PENDING (4-GPU node
  contention) — screen aggregate blocked on them. No code issue.
- **Segmented engine VALIDATED on real B/2** (heun 25677697 nfe/img 747.7; alloc 25677699
  747.0; n=256, COMPLETED). churn/optsw PD (same code, expected pass).
- Two real bugs caught + fixed by the watcher loop (local CPU smoke missed both — fake model
  too simple; per CLAUDE.md smoke-insufficiency rule, now hardened with a Conv layer):
  1. `134b326` — segmented final `feat()` built an autograd graph → `.numpy()` on grad tensor.
     Fix: global `torch.set_grad_enabled(False)`.
  2. `b10cad7` — `eta` float64 → `et` double → `xt` upcast → model float32 Conv bias rejected
     ("Input type double, bias float"). Fix: `et` forced float32.
- Segmented FULL screen (n=10k) NOT launched yet — deliberately held so it doesn't starve the
  priority selection screen of nodes. Launch after selection screen drains.

## Known caveats / honest flags
- `smc_metacog` at keep-1 matched NFE reduces to risk-weighted selection (extra
  particles/jitter not affordable at 750 NFE) — labeled as such, not "full SMC".
- Segmented `heun_corrector` is intentionally OFF-BUDGET (extra corrector evals);
  aggregator flags it; not promotable against the matched baseline without a
  budget-funding variant.
- `cem_elite`/`branch_lookahead`/`oracle_imitation` specced but NOT launched —
  need the extra-draw segmented path hardened (risk too high for unattended run).
- Screen is seed0/n=10k → ranking only; any promoted number is confirmed at 50k×5.
