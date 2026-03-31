# Experimental Design: Adversarial Negative Mining for Matrix Inversion

**Date:** 2026-01-13
**Status:** DEBATE_COMPLETE

---

## Research Question

**Does adversarial negative mining improve performance on the matrix inversion task?**

---

## Experimental Design

### Hypothesis

Gradient-based adversarial negative mining (using the existing `opt_step` method to generate hard negatives) will improve model performance on matrix inversion compared to random perturbation mining or no mining.

### Experimental Conditions

We will compare three training strategies:

1. **Baseline (No Mining)**
   - Standard diffusion training without negative mining
   - Configuration: `supervise_energy_landscape=False`, `use_innerloop_opt=False`

2. **Random Mining**
   - Current IRED approach: negative samples generated via random perturbations
   - Configuration: `supervise_energy_landscape=True`, `use_innerloop_opt=False`, `mining_mode='random'`

3. **Adversarial Mining**
   - Gradient-based hard negative generation using `opt_step`
   - Configuration: `supervise_energy_landscape=True`, `use_innerloop_opt=True`, `mining_mode='adversarial'`
   - Initial parameters: `mining_opt_steps=2`, `mining_scale_factor=1.0`

### Dataset

- **Task:** Matrix inversion (Inverse dataset from dataset.py)
- **Matrix size:** 20×20 (rank=20)
- **Training distribution:** Symmetric positive definite matrices generated via `A = R·R^T + 0.5·I`
- **Training size:** Infinite (procedurally generated)
- **Validation:** Same distribution as training

### Model Architecture

- **Model:** MLP-based energy model (EBM class from models.py)
- **Wrapper:** DiffusionWrapper
- **Diffusion steps:** 10 (default)
- **Input dimension:** 400 (20×20 matrix flattened)
- **Output dimension:** 400 (20×20 inverse matrix flattened)

### Training Configuration

- **Batch size:** 2048
- **Learning rate:** 1e-4
- **Training steps:** 100,000 (quick pilot) → 1,300,000 (full run)
- **Gradient accumulation:** 1
- **EMA decay:** 0.995
- **Mixed precision:** False
- **Optimization:** AdamW

### Evaluation Metrics

1. **Primary metric:** Mean Squared Error (MSE) on validation set
2. **Secondary metrics:**
   - Matrix reconstruction quality: `||A·A_inv - I||_F`
   - Energy landscape quality (for mining conditions)
   - Training loss curves
   - Inference time

### Success Criteria

Adversarial mining is considered successful if:
1. Validation MSE is significantly lower than baseline (>5% improvement)
2. Matrix reconstruction error is lower than baseline
3. Training remains stable (no divergence)
4. Additional computational cost is acceptable (<2× training time)

---

## Implementation Approach

**Decision:** Hybrid approach (Position C from debate)

### Rationale

After structured debate, the hybrid approach was selected because it:
1. Reuses proven infrastructure (Trainer1D, GaussianDiffusion1D)
2. Provides clear experimental narrative through orchestration script
3. Enables easy ablation studies (vary opt_steps, scale_factor, etc.)
4. Minimizes risk of bugs compared to reimplementation
5. Balances speed (3-4 hours setup) with clarity and extensibility

### Architecture

```
projects/ired/
├── train.py                          # Core training infrastructure (unchanged)
├── dataset.py                        # Datasets including Inverse (unchanged)
├── models.py                         # EBM models (unchanged)
├── diffusion_lib/
│   └── denoising_diffusion_pytorch_1d.py  # Modify: add mining_config parameter
└── experiments/
    └── matrix_inversion_mining_comparison.py  # New: orchestration script
```

### Key Modifications

**1. GaussianDiffusion1D (denoising_diffusion_pytorch_1d.py)**

Add `mining_config` parameter to `__init__`:
```python
def __init__(self, ..., mining_config=None):
    self.mining_config = mining_config or {
        'mode': 'random',  # 'none', 'random', 'adversarial'
        'opt_steps': 2,
        'scale_factor': 1.0,
    }
```

Modify `p_losses()` method (lines 605-699) to support configurable mining:
```python
if self.supervise_energy_landscape:
    if self.mining_config['mode'] == 'none':
        # Skip negative mining entirely
        pass
    elif self.mining_config['mode'] == 'random':
        # Current approach: random perturbations (3.0 * noise)
        xmin_noise = self.q_sample(x_start=x_start, t=t, noise=3.0 * noise)
        xmin_noise_rescale = self.predict_start_from_noise(
            xmin_noise, t, torch.zeros_like(xmin_noise)
        )
    elif self.mining_config['mode'] == 'adversarial':
        # Gradient-based hard negatives via opt_step
        xmin_noise = self.q_sample(x_start=x_start, t=t, noise=3.0 * noise)
        xmin_noise = self.opt_step(
            inp, xmin_noise, t, mask, data_cond,
            step=self.mining_config['opt_steps'],
            sf=self.mining_config['scale_factor']
        )
        xmin_noise_rescale = self.predict_start_from_noise(
            xmin_noise, t, torch.zeros_like(xmin_noise)
        )
```

**2. Experiment Orchestration Script (experiments/matrix_inversion_mining_comparison.py)**

```python
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D
from models import EBM, DiffusionWrapper
from dataset import Inverse

def get_mining_config(strategy):
    """Return mining configuration for each strategy"""
    configs = {
        'baseline': {
            'supervise_energy_landscape': False,
            'use_innerloop_opt': False,
            'mining_config': {'mode': 'none'}
        },
        'random': {
            'supervise_energy_landscape': True,
            'use_innerloop_opt': False,
            'mining_config': {'mode': 'random'}
        },
        'adversarial': {
            'supervise_energy_landscape': True,
            'use_innerloop_opt': True,
            'mining_config': {
                'mode': 'adversarial',
                'opt_steps': 2,
                'scale_factor': 1.0
            }
        }
    }
    return configs[strategy]

def run_experiment(strategy, rank=20, batch_size=2048, train_steps=100000):
    """Run single experiment with specified mining strategy"""
    print(f"\n{'='*60}")
    print(f"Training with {strategy} mining strategy")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = Inverse("train", rank, ood=False)

    # Create model
    model = EBM(inp_dim=dataset.inp_dim, out_dim=dataset.out_dim)
    model = DiffusionWrapper(model)

    # Get configuration
    config = get_mining_config(strategy)

    # Create diffusion model
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=32,
        objective='pred_noise',
        timesteps=10,
        sampling_timesteps=10,
        supervise_energy_landscape=config['supervise_energy_landscape'],
        use_innerloop_opt=config['use_innerloop_opt'],
        mining_config=config['mining_config'],
        continuous=True
    )

    # Create trainer
    trainer = Trainer1D(
        diffusion,
        dataset,
        train_batch_size=batch_size,
        validation_batch_size=256,
        train_lr=1e-4,
        train_num_steps=train_steps,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        data_workers=4,
        amp=False,
        metric='mse',
        results_folder=f'results/mining_comparison/{strategy}',
        validation_dataset=dataset,
        save_and_sample_every=1000
    )

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate()

    return results

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--strategies', nargs='+',
                       default=['baseline', 'random', 'adversarial'],
                       choices=['baseline', 'random', 'adversarial'])
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--train-steps', type=int, default=100000,
                       help='Training steps (100K for pilot, 1.3M for full)')
    parser.add_argument('--pilot', action='store_true',
                       help='Run quick pilot with 10K steps')

    args = parser.parse_args()

    if args.pilot:
        args.train_steps = 10000

    # Run experiments
    results = {}
    for strategy in args.strategies:
        results[strategy] = run_experiment(
            strategy,
            rank=args.rank,
            batch_size=args.batch_size,
            train_steps=args.train_steps
        )

    # Compare results
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}\n")

    for strategy, result in results.items():
        print(f"{strategy:20s}: MSE = {result['mse']:.6f}")

if __name__ == '__main__':
    main()
```

---

## Execution Plan

### Phase 1: Implementation (3-4 hours)

1. **Modify GaussianDiffusion1D** (1.5 hours)
   - Add `mining_config` parameter
   - Refactor `p_losses()` to support configurable mining modes
   - Test that existing functionality still works

2. **Create experiment script** (1.5 hours)
   - Implement `get_mining_config()`
   - Implement `run_experiment()`
   - Add command-line interface
   - Add results comparison

3. **Testing** (1 hour)
   - Verify all three strategies run without errors
   - Check that baseline produces expected results
   - Validate that adversarial mining calls `opt_step`

### Phase 2: Pilot Experiments (30 minutes runtime)

Run short training (10K steps) for all three strategies to:
- Validate implementation correctness
- Check for any runtime errors or instabilities
- Get rough sense of whether adversarial mining shows promise
- Tune hyperparameters if needed

```bash
python experiments/matrix_inversion_mining_comparison.py --pilot
```

### Phase 3: Full Experiments (3-6 hours GPU time)

Run full training (100K-1.3M steps) for each strategy:

```bash
# Baseline
python experiments/matrix_inversion_mining_comparison.py \
    --strategies baseline --train-steps 100000

# Random mining
python experiments/matrix_inversion_mining_comparison.py \
    --strategies random --train-steps 100000

# Adversarial mining
python experiments/matrix_inversion_mining_comparison.py \
    --strategies adversarial --train-steps 100000
```

### Phase 4: Analysis and Ablations (as needed)

If adversarial mining shows promise, run ablations:
- Vary `opt_steps`: 1, 2, 5, 10
- Vary `scale_factor`: 0.5, 1.0, 2.0
- Vary matrix size: rank=10, 20, 30

---

## Expected Outcomes

### Scenario 1: Adversarial Mining Succeeds

- Validation MSE significantly lower (>5% improvement)
- Matrix reconstruction error reduced
- Energy landscape better shaped for optimization
- **Next steps:** Run ablations, extend to other tasks, write paper

### Scenario 2: Random Mining Performs Best

- Random perturbations provide sufficient diversity
- Adversarial mining may be overfitting to specific hard negatives
- **Next steps:** Investigate why gradient-based negatives underperform

### Scenario 3: Baseline Performs Best

- Negative mining provides no benefit for matrix inversion
- Task may not benefit from contrastive learning
- **Next steps:** Investigate other regularization approaches

### Scenario 4: Training Instability

- Adversarial mining causes divergence or unstable training
- May need to tune `opt_steps`, `scale_factor`, or loss weighting
- **Next steps:** Debug and adjust hyperparameters

---

## Resources Required

- **Compute:** 1-4 GPUs
- **Time per experiment:** 1-2 hours (for 100K steps)
- **Storage:** ~500MB per checkpoint, ~1.5GB total for 3 strategies
- **Development time:** 3-4 hours implementation + 1 hour testing

---

## Success Metrics for This Design

This experimental design is successful if:
1. All three strategies can be run with minimal code changes
2. Results clearly show performance differences between strategies
3. Ablations can be easily added by modifying `mining_config`
4. Experimental setup is clear enough to include in paper methods section
