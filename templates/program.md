# program.md — Autoresearch Governance
#
# This file governs autonomous experiment iteration for this project.
# The agent reads this file at the start of each iteration to decide
# what to try next. The human writes this file; the agent writes code.
#
# Modeled after Karpathy's autoresearch paradigm:
# hypothesize → code → run → measure → keep/revert → repeat

## Objective

metric: <metric_name>           # e.g., val_mse, accuracy, val_bpb
direction: minimize             # minimize | maximize
eval_command: <command>         # Command that prints "metric: <value>" to stdout
                                # e.g., "python experiments/evaluate.py --config configs/eval.json"
                                # For SLURM jobs: parsed from log output (grep pattern below)
eval_grep: "^<metric_name>:"   # Regex to extract metric from logs (for SLURM jobs)

## Baseline

# The starting metric value. Set after first successful run.
baseline: null
best_so_far: null
best_commit: null

## Constraints

max_runtime_seconds: 300        # Max wall-clock time per experiment (local)
max_slurm_minutes: 60           # Max SLURM job time (cluster)
files_allowed:                  # Files the agent may modify (all others read-only)
  - experiments/*.py
  - configs/*.json
files_readonly:                 # Files the agent must NEVER modify
  - program.md                  # This governance file
  - experiments/evaluate.py     # Evaluation function (immutable)
partition: gpu_test             # Default SLURM partition
pilot_steps: 1000              # Short pilot run for rapid iteration (before full training)
full_steps: 100000             # Full training run (for validated improvements)

## Hypothesis Generation

# How the agent decides what to try next.
# The agent reads results.tsv (all past experiments) and this section
# to form its next hypothesis.

exploration_dimensions:
  - hyperparameters             # Learning rate, batch size, weight decay, etc.
  - architecture                # Model structure, layer count, hidden size, etc.
  - optimization                # Optimizer choice, schedule, gradient clipping, etc.
  - data                        # Augmentation, preprocessing, sampling, etc.
  - regularization              # Dropout, weight decay, early stopping, etc.
  - loss                        # Loss function modifications, weighting, etc.

strategy: |
  1. Start with the most impactful dimension (usually hyperparameters)
  2. Make ONE change per experiment (isolate variables)
  3. If a dimension yields 3 consecutive failures, move to next dimension
  4. Prefer simpler changes over complex ones
  5. Read results.tsv to avoid repeating failed approaches
  6. When stuck, try the opposite of what failed

## Ratchet Rules

# How to decide whether to keep or revert a change.

keep_threshold: 0.0            # Minimum improvement to keep (0.0 = any improvement)
revert_on_regression: true     # Automatically revert if metric worsens
revert_on_crash: true          # Automatically revert if experiment crashes
max_consecutive_failures: 10   # Stop after N consecutive failures (ask human)
simplicity_preference: true    # Prefer simpler code when performance is equivalent
                               # (within keep_threshold of best)

## Termination

# When to stop the autoresearch loop.

max_iterations: 100            # Maximum number of experiments before stopping
target_metric: null            # Stop if metric reaches this value (null = no target)
max_wall_hours: 12             # Maximum wall-clock hours for the entire loop
stop_on_plateau: true          # Stop if no improvement for plateau_window iterations
plateau_window: 15             # Number of iterations without improvement = plateau

## Execution Mode

# How experiments are run.

mode: slurm                    # local | slurm
                               # local: run eval_command directly, 300s timeout
                               # slurm: submit to cluster, poll for results

## Notes

# Add any project-specific notes, constraints, or context here.
# The agent will read this section for additional guidance.

