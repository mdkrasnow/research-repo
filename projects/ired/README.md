# Adversarial Negative Mining for Matrix Inversion

## Research Question
Does adversarial negative mining improve performance on the matrix inversion task?

## Overview
This project investigates whether gradient-based hard negative mining improves the accuracy of diffusion models on matrix inversion tasks compared to baseline training and random negative sampling.

### Key Hypotheses
1. **Baseline**: Standard diffusion training achieves baseline performance
2. **Random negatives**: Random negative matrices provide weak regularization
3. **Adversarial negatives**: Gradient-based hard negatives significantly improve performance by forcing the model to learn better discrimination

## Approach
We use the IRED diffusion-based reasoning framework with the following components:
- **Model**: Energy-based MLP with time conditioning
- **Dataset**: 20×20 symmetric positive definite matrices with computed inverses
- **Diffusion**: 10-step Gaussian diffusion with noise prediction objective
- **Mining strategies**:
  - `none`: No negative mining (baseline)
  - `random`: Random matrices as negatives
  - `adversarial`: Gradient ascent on energy landscape using `opt_step()`

### Design Decision
After structured debate (see `documentation/debate.md`), we chose a **hybrid approach**:
- Reuse proven infrastructure: `Trainer1D`, `GaussianDiffusion1D`, `Inverse` dataset
- Add custom experiment orchestration for clean comparison of mining strategies
- Balance speed (3-4h implementation) with research clarity and extensibility

## Project Structure
```
projects/ired/
├── experiments/
│   └── matrix_inversion_mining.py    # Main experiment script
├── configs/
│   ├── q001_baseline.json            # No negative mining
│   ├── q002_random.json              # Random negatives
│   ├── q003_adversarial.json         # Gradient-based hard negatives
│   └── q004_pilot.json               # Quick debug run
├── slurm/
│   ├── jobs/                         # SLURM submission scripts
│   └── logs/                         # SLURM output/error logs
├── runs/                             # Per-run outputs and ledgers
├── results/                          # Aggregated results
│   └── summary.md
├── documentation/
│   ├── implementation-todo.md        # Development checklist
│   ├── debugging.md                  # Debugging guide
│   ├── queue.md                      # Experiment queue
│   └── debate.md                     # Design decision record
└── .state/
    └── pipeline.json                 # Project execution state
```

## Quick Start

### 1. Local Testing (Debug Run)
```bash
# Activate Python environment
# source venv/bin/activate  # if using virtualenv

# Run quick pilot test (1000 steps, ~5 minutes on GPU)
cd /path/to/research-repo
python projects/ired/experiments/matrix_inversion_mining.py \
    --config projects/ired/configs/q004_pilot.json
```

### 2. Submit to SLURM (Remote Cluster)
```bash
# First, ensure SSH session is established
scripts/cluster/ssh_bootstrap.sh

# Submit pilot test
scripts/cluster/submit.sh projects/ired/slurm/jobs/q004.sbatch

# Check job status
scripts/cluster/status.sh <job_id>

# Fetch logs after completion
scripts/cluster/remote_fetch.sh ired
```

### 3. Run Full Experiments
```bash
# Option A: Use /dispatch command to advance pipeline
/dispatch --project ired

# Option B: Submit specific experiments manually
scripts/cluster/submit.sh projects/ired/slurm/jobs/q001.sbatch  # Baseline
scripts/cluster/submit.sh projects/ired/slurm/jobs/q002.sbatch  # Random
scripts/cluster/submit.sh projects/ired/slurm/jobs/q003.sbatch  # Adversarial
```

## Implementation Status
**Current Phase**: IMPLEMENT

See `documentation/implementation-todo.md` for detailed task list:
- [ ] T1: Modify GaussianDiffusion1D to add `mining_config` parameter
- [ ] T2: Update Trainer1D to pass mining_config
- [ ] T3: Finalize experiment script
- [ ] T4: Validate configuration files
- [ ] T5: Test SLURM integration
- [ ] T6-T7: Run pilot and full experiments

## Key Insights from Codebase Analysis
### Existing Infrastructure (Reusable)
- **Inverse dataset** (`dataset.py:420-465`): Generates symmetric positive definite matrices with computed inverses
- **EBM + DiffusionWrapper** (`models.py:164-215`, `models.py:789-813`): Energy-based model with gradient computation
- **Trainer1D** (`diffusion_lib/denoising_diffusion_pytorch_1d.py:721-1103`): Full training loop with EMA, validation, checkpointing
- **opt_step()** (`diffusion_lib/denoising_diffusion_pytorch_1d.py:373-406`): Gradient-based optimization for adversarial mining

### What Needs Custom Implementation
- **mining_config parameter** in `GaussianDiffusion1D.__init__()` and `p_losses()`
- **Strategy selection logic** in contrastive loss computation (lines 605-699)
- **Experiment orchestration** script with configuration-based strategy switching

## Expected Results
| Mining Strategy | Expected MSE ↓ | Training Time | Computational Overhead |
|-----------------|----------------|---------------|------------------------|
| Baseline (none) | 0.05-0.10 | 2h | Baseline |
| Random negatives | 0.04-0.08 | 2h | +5% |
| Adversarial | 0.02-0.05 | 2-2.5h | +10-20% (due to opt_step) |

**Success criterion**: Adversarial mining achieves ≥20% lower MSE than baseline.

## Resources
- GPU: 1 GPU per experiment
- Memory: 16GB (8GB for pilot)
- Time: 2 hours per full experiment (15 minutes for pilot)
- Partition: `gpu` (adjust in sbatch scripts if different)

## Next Steps
1. Review implementation tasks: `documentation/implementation-todo.md`
2. Complete Task T1 (modify GaussianDiffusion1D)
3. Run pilot experiment Q-004 to validate setup
4. Execute baseline Q-001, then Q-002 and Q-003
5. Analyze results and update `results/summary.md`
6. Consider extensions: OOD evaluation, larger matrices, different diffusion steps

## References
- Existing IRED codebase: `projects/ired/{train.py, dataset.py, models.py}`
- Diffusion library: `projects/ired/diffusion_lib/denoising_diffusion_pytorch_1d.py`
- Original IRED paper: [Link to paper if available]

## Contact
For questions or issues, see `documentation/debugging.md` or consult the implementation todo list.
