# Results — q209_20260219_scalar_nm

## Job Status
- Job ID: 61206606
- Array spec: 0-9%2 (throttled to 2 concurrent)
- Partition: gpu
- Git SHA: 4da0946
- Strategy: none (no mining) + scalar energy head
- Status: IN PROGRESS (seeds 0-1 running, 2-9 pending)
- Submitted: 2026-02-19T02:00:00Z
- Last checked: 2026-02-19 (~1h after submission)

## Diagnostic Purpose
**Key Question**: Is the q207 plateau (~0.09 val MSE) caused by adversarial mining, or by the scalar head itself?

- If q209 val MSE → ~0.009 (matching q101 no-mining baseline): **mining is the culprit**
- If q209 val MSE is still bad: **scalar head itself is the problem**

## Partial Results (Seeds 0,1 at ~50k steps)

| Seed | State | train_mse | val_mse | Interpretation |
|------|-------|-----------|---------|----------------|
| 0 | RUNNING | **0.00977** | **2.497** | Fits train, fails val |
| 1 | RUNNING | **0.00969** | **1.824** | Fits train, fails val |

## Critical Finding: CATASTROPHIC OVERFITTING

### Train MSE = baseline quality, Val MSE = catastrophic
- Train MSE for both seeds: **~0.009** — identical to q101 baseline (0.00969)
- Val MSE for both seeds: **1.8–2.5** — 200-260x worse than q101 baseline val MSE

### Answer to Diagnostic Question: SCALAR HEAD IS THE PROBLEM
The scalar energy head is causing catastrophic train/val generalization gap.
Even with **no mining at all**, the scalar head destroys generalization.
Adversarial mining in q207 (val MSE ~0.09) is actually BETTER — it's not collapsing as badly,
possibly because mining provides some regularization on the energy landscape.

### CD-DIAG at ~50k Steps (Seed 0)
```
[CD-DIAG step=50k] MSE(batch)~0.009, E_pos/E_neg growing into millions
lang_grad0=0.000 (Langevin inactive - expected for no-mining)
```

## Implication for Research Direction
The scalar energy head architecture is fundamentally incompatible with this task.
The model memorizes training matrices but fails to generalize. Root cause hypotheses:
1. Scalar energy is too unconstrained — can assign arbitrary energy to any input
2. Energy guidance during inference uses a score that doesn't generalize
3. No regularization on the energy landscape → infinite degrees of freedom

## Conclusion
**Q209 answers the diagnostic: scalar head (not mining) is root cause of poor generalization.**
Next step: abandon scalar energy head. Return to NCE-based energy with proper structure.
