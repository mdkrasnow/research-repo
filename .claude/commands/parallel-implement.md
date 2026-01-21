---
description: Parallelize independent TODO items within one project using up to 5 concurrent subagents.
allowed-tools: Read, Write, Edit, Task, Bash(mkdir:*)
argument-hint: --project <slug>
---
# /parallel-implement

Accelerate implementation by running up to 5 independent tasks in parallel within a single project.

## Usage

```bash
/parallel-implement --project my-experiment
```

## What It Does

1. **Analyzes** independent TODO items in `documentation/implementation-todo.md`
2. **Groups** tasks that can run in parallel (max 5 concurrent subagents)
3. **Launches** subagents for each group
4. **Consolidates** results back into documentation and pipeline state
5. **Updates** `implementation-todo.md`, `debugging.md`, and `pipeline.json`

## Output

For each parallel batch:

- Summary of tasks started (up to 5 per wave)
- Task completion status (success/failure)
- Updated documentation files
- Modified entries in `pipeline.json`

## Error Handling

- If a task fails, other parallel tasks complete and remaining non-dependent tasks continue
- Failed task details are logged to `documentation/debugging.md`
- `pipeline.json` updated with partial completion state

## Related Skills

- **`/dispatch`**: Main orchestrator that handles all project phases
- **`/check-status`**: Monitor progress of parallel tasks
- **`/make-project`**: Create project structure with implementation tasks
