---
description: Create a new research project with full structure based on codebase analysis and research question
allowed-tools: Read, Write, Edit, Glob, Grep, Task, Bash(mkdir:*), AskUserQuestion
argument-hint: "<research-question>" --slug <project-slug>
---
# /make-project

Create a complete research project structure by:
1. Understanding the research question through structured questioning
2. Analyzing the existing codebase to identify relevant patterns and code
3. Using internal debate to determine optimal experiment design
4. Generating all required files, configs, and documentation per the project contract

## Usage

```bash
/make-project "Does technique X improve Y?" --slug my-experiment
```

## Process

### Phase 1: Codebase Analysis
Launch parallel exploration tasks to understand relevant code:

1. **Search for relevant experiment patterns**
   - Use Task(Explore) to find existing experiment scripts
   - Identify common patterns (argparse usage, config loading, logging)
   - Find relevant imports and dependencies

2. **Identify baseline code**
   - Search for models, training loops, or relevant algorithms
   - Find config examples that can be adapted
   - Locate utility functions (data loading, metrics, visualization)

3. **Understand execution patterns**
   - Find existing SLURM scripts to understand resource allocation
   - Check how experiments log results
   - Identify output formats and result processing

### Phase 2: Question Refinement
Use AskUserQuestion to gather critical context. Your questions are up to you to choose what you want clarifications you need from the user.  


### Phase 3: Internal Debate - Project Structure Design

Using the gathered information, conduct a structured debate on:

**Position A: Minimal Adaptation**
- Reuse existing experiment scripts with minimal modification
- Copy-paste proven patterns from other projects
- Pros: Fast setup, proven to work
- Cons: May not fit research question perfectly, technical debt

**Position B: Custom Design**
- Design experiment structure tailored to specific research question
- Create new abstractions for clear, reproducible experiments
- Pros: Clean, purpose-built, easier to extend
- Cons: More upfront work, potential for bugs

**Position C: Hybrid Approach**
- Reuse core infrastructure (data loading, training)
- Create custom experiment logic and configs
- Pros: Balance speed and quality
- Cons: Requires careful interface design

**Decision Criteria:**
- Complexity of research question
- Time constraints
- Likelihood of iteration/extension
- Similarity to existing projects

### Phase 4: File Generation

Based on debate outcome and user input, create:

#### Required Directories
```bash
mkdir -p projects/<slug>/.state/lock
mkdir -p projects/<slug>/documentation
mkdir -p projects/<slug>/experiments
mkdir -p projects/<slug>/configs
mkdir -p projects/<slug>/slurm/jobs
mkdir -p projects/<slug>/slurm/logs
mkdir -p projects/<slug>/runs
mkdir -p projects/<slug>/results
```

#### Required Files

**1. `.state/pipeline.json`**
```json
{
  "project": "<slug>",
  "phase": "IMPLEMENT",
  "next_action": {
    "command": "/dispatch",
    "reason": "Begin implementation of experiment infrastructure",
    "hint": "First task: create baseline experiment script"
  },
  "needs_user_input": { "value": false, "prompt": "" },
  "active_runs": [],
  "slurm_wait": { "next_poll_after": null, "poll_interval_seconds": 900 },
  "events": [{
    "id": "evt-0001",
    "ts": "<current-iso-timestamp>",
    "action": "INIT",
    "detail": "Project created via /make-project"
  }]
}
```

**2. `documentation/implementation-todo.md`**
Generate task list based on debate outcome and research needs:
```markdown
# Implementation TODO — <Project Name>

## Ready
- [ ] (T1) Create baseline experiment script in experiments/
- [ ] (T2) Design config schema and create initial configs
- [ ] (T3) Set up SLURM sbatch template with appropriate resources
- [ ] (T4) Implement data loading/preprocessing (if needed)
- [ ] (T5) Implement metrics/evaluation logic
- [ ] (T6) Add result logging and artifact saving
- [ ] (T7) Create initial queue entries for pilot runs
- [ ] (T8) Test local execution with small config

## Blocked
(none)

## Completed
(none yet)
```

**3. `documentation/debugging.md`**
```markdown
# Debugging — <Project Name>

## Active Issues
(none yet)

## Resolved
(none yet)

---
## Common Failure Modes (Preemptive Checklist)
- [ ] Import errors (missing dependencies)
- [ ] Config format mismatches
- [ ] Path errors (relative vs absolute)
- [ ] Resource limits (OOM, timeout)
- [ ] SLURM module loading issues
```

**4. `documentation/queue.md`**
Generate initial queue based on research question:
```markdown
# Experiment Queue — <Project Name>

## Research Question
<user-provided question>

## Hypotheses to Test
<derived from user input and debate>

---

## READY

### Q-001: Baseline
- Hypothesis: Establish baseline performance with default settings
- Config: `configs/q001_baseline.json`
- Resources: <from user input>
- Priority: HIGH (must run first)
- Notes: Foundation for all comparisons

### Q-002: <First variation>
- Hypothesis: <specific prediction>
- Config: `configs/q002_<variant>.json`
- Resources: <from user input>
- Notes: <any dependencies or special considerations>

## IN_PROGRESS
(none)

## DONE
(none)

## FAILED
(none)
```

**5. `results/summary.md`**
```markdown
# Results Summary — <Project Name>

## Research Question
<user-provided question>

## Key Findings
- TBD (will update as results come in)

## Completed Experiments
(none yet)

## Next Steps
- Complete implementation tasks
- Run baseline (Q-001)
- Compare variations
```

**6. Initial experiment script: `experiments/run_experiment.py`**
Create a template based on codebase patterns found:
```python
#!/usr/bin/env python3
"""
Experiment runner for: <Project Name>

Usage:
    python experiments/run_experiment.py --config configs/<name>.json
"""

import argparse
import json
from pathlib import Path
import sys

# Add repo root to path if needed
# sys.path.insert(0, str(Path(__file__).parent.parent))

def load_config(config_path):
    """Load experiment configuration."""
    with open(config_path) as f:
        return json.load(f)

def run_experiment(config):
    """
    Main experiment logic.

    TODO: Implement based on research question:
    - Load data
    - Set up model/system
    - Run experiment
    - Compute metrics
    - Save results
    """
    print(f"Running experiment with config: {config}")

    # Placeholder - replace with actual implementation
    results = {
        "status": "completed",
        "metrics": {},
        "notes": "Placeholder implementation"
    }

    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to config JSON')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Run experiment
    results = run_experiment(config)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir / 'results.json'}")

if __name__ == '__main__':
    main()
```

**7. Initial configs: `configs/q001_baseline.json`**
```json
{
  "experiment_name": "q001_baseline",
  "description": "Baseline configuration",
  "seed": 42,
  "parameters": {
    "_comment": "Add relevant parameters based on research question"
  },
  "output": {
    "save_artifacts": true,
    "log_interval": 100
  }
}
```

**8. SLURM template: `slurm/jobs/q001.sbatch`**
```bash
#!/bin/bash
#SBATCH --job-name=<slug>_q001
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=<time-based-on-user-input>
#SBATCH --<partition-based-on-user-input>
# Example: --partition=cpu or --partition=gpu --gres=gpu:1

set -e

# Navigate to repo root
cd /path/to/research-repo  # This will be updated by /run-experiments

# Load modules if needed
# module load python/3.9
# module load cuda/11.7

# Run experiment
python experiments/run_experiment.py \
    --config configs/q001_baseline.json \
    --output-dir runs/<run-id-will-be-set>

echo "Experiment completed at $(date)"
```

**9. `README.md`**
```markdown
# <Project Name>

## Research Question
<user-provided question>

## Approach
<summary from debate outcome>

## Project Structure
- `experiments/`: Experiment runner scripts
- `configs/`: Experiment configurations (JSON)
- `slurm/`: SLURM submission scripts and logs
- `runs/`: Individual run outputs and ledgers
- `results/`: Aggregated results and summaries
- `documentation/`: Implementation tasks, debugging, experiment queue

## Quick Start

### Local Testing
```bash
python experiments/run_experiment.py --config configs/q001_baseline.json
```

### Submit to SLURM
```bash
/dispatch  # or /run-experiments if implementation complete
```

## Status
See `documentation/queue.md` for experiment status and `.state/pipeline.json` for pipeline state.
```

### Phase 5: Final Report

After creation, report to user:

```markdown
## Project Created: projects/<slug>/

### Research Question
<question>

### Recommended First Steps
1. Review implementation tasks: `documentation/implementation-todo.md`
2. Customize experiment script: `experiments/run_experiment.py`
3. Update configs: `configs/q001_baseline.json`
4. Run `/dispatch` to begin implementation phase

### Key Insights from Codebase Analysis
- <what patterns were found>
- <what code can be reused>
- <what needs custom implementation>

### Debate Outcome
<which approach was chosen and why>

### Next Command
`/dispatch --project <slug>` to start implementation
```

## Implementation Notes

1. **Parallelization**: Run codebase exploration tasks in parallel using multiple Task(Explore) agents
2. **Debate Efficiency**: Use a single Task(general-purpose) agent to conduct internal debate with positions A/B/C
3. **Smart Defaults**: Use user answers to set:
   - SLURM time limits (1h for quick, 12h for long)
   - Partition selection (cpu vs gpu)
   - Number of initial queue items (2-3 for simple, 5+ for sweeps)
   - Experiment script complexity
4. **Validation**: After creation, verify all required files exist and pipeline.json is valid JSON
5. **Git**: Optionally stage new files (but don't commit without user approval)

## Error Handling

- If slug already exists: ask user to confirm overwrite or choose new name
- If codebase search finds nothing relevant: use minimal template with warnings
- If user cancels during questions: abort gracefully without creating partial structure
- If debate cannot reach consensus: default to hybrid approach with explanation
