# Debate: Optimal Project Structure for Adversarial Negative Mining Research

**Date:** 2026-01-13
**Research Question:** Does adversarial negative mining improve performance on the matrix inversion task?

---

## Round 1: Initial Positions

### Position A: Minimal Adaptation (Reuse existing train.py)

**Advocate:** Pragmatic Engineer

**Argument:**
The existing train.py provides a battle-tested training infrastructure with 1.3M training steps, proper validation, checkpointing, and metrics. Adding new command-line flags (e.g., `--negative-mining-strategy`, `--mining-steps`, `--mining-scale-factor`) would be the fastest path to results.

**Pros:**
1. **Fastest time to first results**: Leverages existing Trainer1D, GaussianDiffusion1D, and all supporting infrastructure
2. **Proven stability**: train.py has been used successfully for multiple tasks (addition, lowrank, inverse)
3. **Minimal code changes**: Only need to modify the negative mining section (lines 605-699 in denoising_diffusion_pytorch_1d.py)
4. **Easy comparison**: Can switch between strategies with a single flag change

**Cons:**
1. **Cluttered codebase**: Adding more flags to an already complex train.py (318 lines with many conditionals)
2. **Unclear experimental narrative**: train.py handles 15+ different datasets; matrix inversion is just one of many
3. **Difficult to isolate effects**: Hard to ensure that only the negative mining strategy is changing between runs
4. **Limited extensibility**: Adding more ablations (e.g., different mining frequencies, ensemble strategies) will further complicate train.py

**Proposed Implementation:**
```python
# In train.py
parser.add_argument('--negative-mining-strategy', type=str,
                   choices=['none', 'random', 'adversarial'], default='random')
parser.add_argument('--mining-opt-steps', type=int, default=2)
parser.add_argument('--mining-scale-factor', type=float, default=1.0)

# Pass to diffusion model
diffusion = GaussianDiffusion1D(
    model,
    negative_mining_strategy=FLAGS.negative_mining_strategy,
    mining_opt_steps=FLAGS.mining_opt_steps,
    mining_scale_factor=FLAGS.mining_scale_factor,
    ...
)
```

**Estimated time to first pilot:** 1-2 hours

---

### Position B: Custom Experiment Script

**Advocate:** Research Scientist

**Argument:**
Research clarity demands a clean, focused experimental setup. A custom script (e.g., `experiments/matrix_inversion_mining.py`) that explicitly compares baseline vs adversarial mining provides the clearest scientific narrative and easiest path to publication-quality results.

**Pros:**
1. **Crystal-clear experimental design**: Script explicitly shows what's being compared
2. **Self-documenting**: Code directly reflects the research question
3. **Easy to understand results**: Output clearly maps to experimental conditions
4. **Publication-ready**: Clean separation makes it easy to write methods section
5. **Full control**: Can customize training loop, logging, and metrics specifically for this research question

**Cons:**
1. **Code duplication**: Will need to reimplement or copy training boilerplate from Trainer1D
2. **Risk of subtle bugs**: Reimplementing training logic might introduce inconsistencies
3. **Longer initial setup**: Need to build custom training loop, validation, checkpointing
4. **Maintenance burden**: Changes to core infrastructure won't automatically propagate

**Proposed Implementation:**
```python
# experiments/matrix_inversion_mining.py
def train_baseline(model, dataset, config):
    """Train with no negative mining"""
    ...

def train_random_mining(model, dataset, config):
    """Train with random perturbation mining"""
    ...

def train_adversarial_mining(model, dataset, config):
    """Train with gradient-based hard negative mining"""
    ...

def compare_strategies(config):
    """Run all three strategies and compare"""
    results = {}
    for strategy in ['baseline', 'random', 'adversarial']:
        model = create_model(config)
        results[strategy] = train_and_evaluate(model, strategy, config)
    return results
```

**Estimated time to first pilot:** 4-6 hours (including boilerplate development)

---

### Position C: Hybrid Approach

**Advocate:** Systems Architect

**Argument:**
The optimal solution reuses core infrastructure (Trainer1D, GaussianDiffusion1D, dataset classes) while creating a lightweight, focused experiment orchestration script. This provides both speed and clarity by separating concerns: infrastructure vs experimental design.

**Pros:**
1. **Best of both worlds**: Fast development + clear experimental narrative
2. **Modular and extensible**: Easy to add new mining strategies or ablations
3. **Leverages existing code**: Uses proven Trainer1D, just with different configurations
4. **Clean experimental structure**: Experiment script shows what varies between conditions
5. **Low maintenance**: Core infrastructure changes automatically benefit experiments

**Cons:**
1. **Requires careful interface design**: Need to ensure diffusion model supports all necessary mining configurations
2. **Initial architectural overhead**: Must design clean configuration system
3. **Potential for abstraction leakage**: If Trainer1D doesn't expose right hooks, might need modifications

**Proposed Implementation:**
```python
# experiments/matrix_inversion_mining_comparison.py
from train import create_diffusion_model, create_dataset
from diffusion_lib.denoising_diffusion_pytorch_1d import Trainer1D

def get_mining_config(strategy):
    """Return configuration for each mining strategy"""
    configs = {
        'baseline': {
            'supervise_energy_landscape': False,
            'use_innerloop_opt': False
        },
        'random': {
            'supervise_energy_landscape': True,
            'use_innerloop_opt': False,
            'mining_mode': 'random'
        },
        'adversarial': {
            'supervise_energy_landscape': True,
            'use_innerloop_opt': True,
            'mining_mode': 'adversarial',
            'mining_opt_steps': 2,
            'mining_scale_factor': 1.0
        }
    }
    return configs[strategy]

def run_experiment(strategy, rank=20, batch_size=2048, train_steps=100000):
    """Run single experiment with specified mining strategy"""
    config = get_mining_config(strategy)
    dataset = create_dataset('inverse', rank, ood=False)
    model = create_diffusion_model('mlp', dataset, **config)

    trainer = Trainer1D(
        model,
        dataset,
        train_batch_size=batch_size,
        train_num_steps=train_steps,
        results_folder=f'results/mining_comparison/{strategy}',
        ...
    )

    trainer.train()
    return trainer.evaluate()

if __name__ == '__main__':
    strategies = ['baseline', 'random', 'adversarial']
    results = {}

    for strategy in strategies:
        print(f"\n{'='*60}\nTraining with {strategy} mining\n{'='*60}")
        results[strategy] = run_experiment(strategy)

    # Compare results
    compare_and_visualize(results)
```

**Estimated time to first pilot:** 2-3 hours (refactoring + experiment script)

---

## Round 2: Critical Analysis and Rebuttal

### Position A Response to Criticisms

**Rebuttal by Pragmatic Engineer:**

The concerns about "cluttering" train.py are overstated. The script already successfully handles 15+ datasets with clear conditional logic. Adding 2-3 flags for mining strategy is a marginal increase in complexity. More importantly, the question is: *what does the user need first?*

The user wants "quick experiments (1-2 hours per run)" and needs to validate the research direction. Position A delivers results in 1-2 hours of development time. Positions B and C introduce 2-4+ hours of upfront overhead before the first result.

**Counter-argument to Position B:**
Position B's "research clarity" argument conflates *code structure* with *experimental clarity*. Clear experiments come from good logging, proper metrics, and thoughtful design - not from file organization. A well-commented train.py with clear flag documentation is just as understandable as a custom script.

**Counter-argument to Position C:**
Position C's "hybrid approach" sounds appealing but requires architectural changes that may uncover unexpected dependencies. The Trainer1D class tightly couples training logic with validation, checkpointing, and logging. Creating a clean configuration interface might require modifying Trainer1D internals, adding hidden complexity.

**Strengthened position:**
For a quick pilot study, minimize upfront investment. After validating that adversarial mining *actually works*, invest in cleaner architecture for the full study.

---

### Position B Response to Criticisms

**Rebuttal by Research Scientist:**

The "code duplication" concern misses the point. Research code should prioritize *clarity of scientific narrative* over DRY principles. A 200-line custom experiment script that clearly shows the experimental design is worth more than reusing a 318-line train.py where the experiment is buried in conditionals.

**Counter-argument to Position A:**
Position A underestimates the difficulty of "just adding flags." The existing negative mining code (lines 605-699) has special cases for sudoku, connectivity, and shortest_path tasks. Modifying this section to support configurable mining strategies requires careful refactoring to avoid breaking existing functionality. This is hidden complexity.

**Counter-argument to Position C:**
Position C's "hybrid approach" still requires learning and navigating the Trainer1D API. If we're going to invest time in understanding the training infrastructure anyway, why not build exactly what we need rather than working within existing constraints?

**Strengthened position:**
The fastest path to *understanding* is a simple, self-contained experiment. Copy the core training loop (50-100 lines), implement three clear functions for each strategy, and run. No hidden dependencies, no surprising interactions, no "but it works for the other 15 datasets" debugging.

---

### Position C Response to Criticisms

**Rebuttal by Systems Architect:**

Both Position A and B miss the key insight: the user will want to run *multiple ablations* quickly. This isn't a one-shot experiment - it's the start of a research thread that will require:
- Different mining step counts (1, 2, 5, 10 steps)
- Different scale factors (0.5, 1.0, 2.0)
- Different mining frequencies (every batch, every N batches)
- Ensemble approaches (combine random + adversarial)

Position A's flag approach will explode into 10+ flags. Position B's custom script will require duplicating the ablation logic across multiple functions.

**Counter-argument to Position A:**
The 1-2 hour estimate for Position A is optimistic. Modifying the complex conditional logic in lines 605-699 without breaking sudoku/connectivity/shortest_path requires careful testing. More realistically: 2-3 hours development + 1-2 hours debugging edge cases.

**Counter-argument to Position B:**
Position B's 4-6 hour estimate doesn't account for validation infrastructure. The Trainer1D class provides:
- Validation dataset evaluation with proper metrics
- Extra validation datasets at different intervals
- Checkpoint saving and loading
- Learning rate scheduling
- Gradient accumulation
- EMA model tracking

Reimplementing all of this is not 4-6 hours - it's 1-2 days of careful development.

**Strengthened position:**
The hybrid approach requires 2-3 hours upfront but saves 5-10+ hours across the full research trajectory. Create helper functions in train.py to build configured models, then write a simple orchestration script that calls these helpers. This gives clarity without duplication.

---

## Round 3: Synthesis and Decision

### Evaluation Against Decision Criteria

**1. Research Clarity: Can we easily understand the effect of negative mining?**

- Position A: Medium. Requires examining flags and tracing through conditionals
- Position B: High. Explicit functions for each strategy make design obvious
- Position C: High. Configuration dictionaries + orchestration script clearly show what varies

Winner: Tie between B and C

**2. Time to First Results: How quickly can we run pilot experiments?**

- Position A: ~2-3 hours (development + debugging)
- Position B: ~6-8 hours (development + reimplementation + debugging)
- Position C: ~3-4 hours (refactoring + experiment script + testing)

Winner: Position A (with Position C close second)

**3. Reproducibility: Are experiments well-documented and repeatable?**

- Position A: Medium. Requires documenting flag combinations
- Position B: High. Code is self-documenting
- Position C: High. Configuration dicts + version control provide clear record

Winner: Tie between B and C

**4. Extensibility: Can we easily add more ablations or mining strategies?**

- Position A: Low. Each ablation = more flags and conditionals
- Position B: Medium. Each ablation = duplicate training logic with variations
- Position C: High. Each ablation = new config dict, same orchestration

Winner: Position C

---

### Final Recommendation

**DECISION: Adopt Position C (Hybrid Approach) with pragmatic simplifications**

**Justification:**

After three rounds of debate, Position C emerges as the optimal choice when we account for the full research trajectory:

1. **Time to first results**: Position C's 3-4 hour timeline is acceptable given the user's 1-2 hour *experiment* runtime. The small upfront investment pays dividends immediately.

2. **Research clarity**: Position C matches Position B's clarity while maintaining Position A's speed through code reuse.

3. **Extensibility**: The user's research question ("does adversarial negative mining improve performance") will naturally lead to follow-up questions:
   - How many opt_steps are optimal?
   - What scale factor works best?
   - Does mining frequency matter?

   Position C makes these ablations trivial; Position A makes them painful.

4. **Reduced risk**: Position C leverages proven infrastructure (Trainer1D) while Position B risks subtle bugs in reimplementation. Position A risks breaking existing functionality.

**Implementation Plan:**

1. **Phase 1: Minimal refactoring (30 minutes)**
   - Extract model/dataset creation into helper functions in train.py
   - Add `mining_config` parameter to GaussianDiffusion1D initialization
   - Verify existing functionality still works

2. **Phase 2: Experiment script (1.5 hours)**
   - Create `experiments/matrix_inversion_mining_comparison.py`
   - Implement `get_mining_config()` for baseline/random/adversarial
   - Implement `run_experiment()` using Trainer1D
   - Add results comparison and visualization

3. **Phase 3: Initial pilot (1 hour)**
   - Run short training (10K steps instead of 1.3M) for all three strategies
   - Validate that adversarial mining shows different behavior
   - Iterate on mining parameters if needed

4. **Phase 4: Full experiments (1-2 hours each)**
   - Run full training for each strategy
   - Evaluate on test set
   - Compare performance metrics

**Key modifications needed:**

```python
# In denoising_diffusion_pytorch_1d.py GaussianDiffusion1D.__init__()
def __init__(self, ..., mining_config=None):
    self.mining_config = mining_config or {
        'mode': 'random',  # 'none', 'random', 'adversarial'
        'opt_steps': 2,
        'scale_factor': 1.0,
    }

# In p_losses() method, replace lines 657-677 with:
if self.mining_config['mode'] == 'none':
    # No negative mining - just standard diffusion loss
    xmin_noise_rescale = None
elif self.mining_config['mode'] == 'random':
    # Current random perturbation approach
    xmin_noise = self.q_sample(x_start=x_start, t=t, noise=3.0 * noise)
    xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t, torch.zeros_like(xmin_noise))
elif self.mining_config['mode'] == 'adversarial':
    # Gradient-based hard negative mining
    xmin_noise = self.q_sample(x_start=x_start, t=t, noise=3.0 * noise)
    xmin_noise = self.opt_step(
        inp, xmin_noise, t, mask, data_cond,
        step=self.mining_config['opt_steps'],
        sf=self.mining_config['scale_factor']
    )
    xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t, torch.zeros_like(xmin_noise))
```

**Why this wins:**

- Faster than Position B (reuses infrastructure)
- Clearer than Position A (explicit experimental design)
- More extensible than both (configuration-based)
- Lower risk than B (uses proven Trainer1D)
- More maintainable than A (clean separation of concerns)

**Total estimated time investment:**
- Development: 3-4 hours
- First pilot (10K steps): 15-30 minutes
- Full experiments (3 strategies × 100K steps): 3-6 hours GPU time

**Next steps:**
1. Create configuration system in GaussianDiffusion1D
2. Write experiment orchestration script
3. Run quick pilot to validate approach
4. Execute full experimental comparison
