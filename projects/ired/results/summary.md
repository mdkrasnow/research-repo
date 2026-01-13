# Results Summary — Adversarial Negative Mining for Matrix Inversion

## Research Question
Does adversarial negative mining improve performance on the matrix inversion task?

## Key Findings
- TBD (will update as results come in)

---

## Completed Experiments
(none yet)

---

## Planned Comparisons

### Primary Comparison: Mining Strategy Effect
| Experiment | Mining Strategy | MSE ↓ | Training Time | Energy Quality |
|------------|----------------|-------|---------------|----------------|
| Q-001 | Baseline (none) | TBD | TBD | TBD |
| Q-002 | Random negatives | TBD | TBD | TBD |
| Q-003 | Adversarial (gradient-based) | TBD | TBD | TBD |

**Expected outcome**: Q-003 (adversarial) should achieve lowest MSE and best energy discrimination.

### Secondary Analysis
- **Convergence speed**: Steps required to reach MSE threshold (e.g., 0.01)
- **Sample efficiency**: Performance at 25K, 50K, 75K, 100K steps
- **Computational overhead**: Adversarial mining adds opt_step cost (expect ~10-20% slowdown)

---

## Next Steps
1. Complete implementation tasks (see `documentation/implementation-todo.md`)
2. Run pilot experiment Q-004 to validate setup
3. Execute baseline Q-001 first
4. Run Q-002 and Q-003 in parallel (independent experiments)
5. Analyze results and update this summary
6. Consider extensions: OOD evaluation, larger matrices, diffusion step ablations

---

## Analysis Plan
Once experiments complete:
1. **Quantitative comparison**: MSE, per-element accuracy, convergence curves
2. **Energy landscape visualization**: Compare energy distributions for positives vs negatives
3. **Statistical testing**: Paired t-test or Wilcoxon signed-rank test for significance
4. **Qualitative analysis**: Inspect failure cases (matrices with high error)

---

## Notes
- All experiments use 20×20 matrices (rank=20), 10 diffusion steps, 100K training steps
- Results stored in `runs/<run_id>/results.json` with checkpoints at `results/ds_inverse/model_mlp/`
