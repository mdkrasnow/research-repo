# Reproducibility

Every run must create:
- `runs/<run_id>/spec.md`
- `runs/<run_id>/config_snapshot.*`
- `runs/<run_id>/submit.json`
- `runs/<run_id>/results.md`

`submit.json` must include: job id(s), command(s), resources, timestamps, env notes.
