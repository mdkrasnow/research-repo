# Handwritten notes (transcribed summary)

The notes describe a research automation loop:

- Use slash commands to:
  - implement tasks
  - run adversarial tests
  - submit experiments
  - check results
  - debate decisions when necessary
- Maintain per-project docs:
  - experimental design
  - implementation TODO
  - debugging
  - experiment queue
- Maintain per-project state (`pipeline.json`) so projects don’t collide.
- SLURM flow:
  - create a run ledger and record job IDs
  - wait for jobs to complete before checking results
  - link results back to queue entries and debugging
