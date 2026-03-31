# Results — q222_tam_delta_sweep

## Job Status
- Job ID: 61733316
- Status: COMPLETED
- Submitted: 2026-02-23 09:15:00 UTC
- Duration: ~1.5 hours

## Experiment
- **Name**: TAM pgd_delta hyperparameter sweep
- **Configurations**: 4 configs × 4 seeds = 16 seeds total
  - Delta values: 0.5, 1.0 (baseline), 2.0, 3.0
  - All configs use anchor_step=1 (from q220)

## Results Summary

### Validation MSE
- **Mean**: 0.00972790
- **Std**: 0.00001570
- **Min**: 0.00969548 (Seed 11)
- **Max**: 0.00975175 (Seed 15)
- **Stability**: Excellent (coefficient of variation = 0.16%)

### Key Findings
1. **Delta Parameter Insensitive**: Validation MSE shows virtually no dependence on pgd_delta
2. **Comparison to Q220 TAM Baseline**:
   - Q220: 0.00972801 ± 0.00001577 (8 seeds)
   - Q222: 0.00972790 ± 0.00001570 (16 seeds)
   - Difference: -0.00000011 (negligible)
   - Suggests delta is not a critical hyperparameter for TAM

### Per-Seed Results
```
Seed 0:  0.0097269
Seed 1:  0.00973049
Seed 2:  0.00974756
Seed 3:  0.0097281
Seed 4:  0.00972724
Seed 5:  0.00973075
Seed 6:  0.00974943
Seed 7:  0.00972757
Seed 8:  0.00972626
Seed 9:  0.00973074
Seed 10: 0.00970446
Seed 11: 0.00969548 (best)
Seed 12: 0.00970415
Seed 13: 0.00972825
Seed 14: 0.00973722
Seed 15: 0.00975175
```

## Analysis

### Delta Impact (Minimal)
The pgd_delta parameter appears to have negligible effect on validation performance across the tested range (0.5-3.0). This suggests:
- TAM's core mechanism (anchor-based negative selection) is robust to delta variation
- Delta might primarily affect training dynamics rather than final performance
- No need for aggressive delta tuning

### Next Steps
Since q222 shows delta insensitivity:
1. **q221 (anchor_step sweep)**: Currently running, will test anchor_step variation
2. **Interpretation**: Combine q221 results with q222 to identify truly sensitive hyperparameters
3. **Decision**: May skip other hyperparameter sweeps if q221 is also insensitive

## SLURM Details
- Partition: gpu (multiple runs across partitions due to array scheduling)
- Array: 0-15
- Script: `projects/ired/slurm/q222.sbatch`
- Git SHA: 8771f66
