# Workflow

1. Define/clarify the research question and success criteria
2. Design experiments
3. Implement
4. Test (incl. adversarial)
5. Run experiments (SLURM)
6. Check results
7. Debug/iterate
8. Ship findings

- `/dispatch` is the main orchestrator.
- The **Ralph loop** keeps Claude going by blocking Stop when work is actionable.

Per project:
- state: `projects/<slug>/.state/pipeline.json`
- docs: `documentation/implementation-todo.md`, `debugging.md`, `queue.md`
- outputs: `runs/<run_id>/...` and `results/`
