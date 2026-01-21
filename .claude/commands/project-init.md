---
description: Initialize a new project directory with standard project template structure.
allowed-tools: Read, Write, Bash, Glob
argument-hint: --project <slug>
---
# /project-init

Create a new project directory from the standard template with all required structure.

## Usage

```bash
/project-init --project my-experiment          # Create new project
/project-init --project baseline-comparison    # Create another project
```

## What It Does

1. **Validates** project slug (alphanumeric, hyphens only)
2. **Copies** template from `templates/project/` to `projects/<slug>/`
3. **Replaces** all `<slug>` placeholders with actual project slug
4. **Initializes** `pipeline.json` with INIT phase
5. **Registers** project in `projects/PROJECTS.md`

## Directory Structure Created

```
projects/<slug>/
├── .state/
│   ├── pipeline.json          # Pipeline state & tracking
│   └── lock/                  # Locking directory
├── documentation/
│   ├── implementation-todo.md # Task list
│   ├── debugging.md           # Debugging log
│   └── queue.md               # Experiment queue
├── experiments/               # Experiment scripts
├── configs/                   # Config files
├── slurm/
│   ├── jobs/                  # SLURM sbatch scripts
│   └── logs/                  # SLURM output logs
├── runs/                      # Individual run outputs
├── results/                   # Aggregated results
└── README.md                  # Project documentation
```

## Output

```
✓ Project initialized
Created: projects/my-experiment/
Phase: INIT
Next: Use /make-project for structured setup or /dispatch to begin
```

## Error Handling

- **Slug already exists**: Returns error "Project already exists"
- **Invalid slug format**: Returns error "Invalid slug: must be alphanumeric with hyphens"
- **Template missing**: Returns error "Template not found in templates/project/"

## Related Skills

- **`/make-project`**: Structured project creation with questions
- **`/dispatch`**: Start project through its pipeline phases
- **`/check-status`**: Monitor initialized projects
