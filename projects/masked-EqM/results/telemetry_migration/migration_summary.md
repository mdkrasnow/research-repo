# Telemetry migration summary

Migrator version 1; source `.`; output `results/telemetry`.

## Totals

| quantity | value |
| --- | ---: |
| legacy_facts | 1446 |
| attributed_facts | 1397 |
| unattributed_facts | 49 |
| reconstructed_runs | 961 |
| reconstructed_executions | 993 |
| metric_streams | 9 |
| metric_streams_with_job_id | 0 |
| events_emitted | 8657 |

## Events by type

| key | count |
| --- | ---: |
| `PROGRESS` | 5120 |
| `OBSERVED` | 1406 |
| `NOTICE` | 1273 |
| `END` | 858 |

## PROGRESS records by kind

| key | count |
| --- | ---: |
| `grad` | 4200 |
| `wfb` | 920 |

## Runs by confidence

| key | count |
| --- | ---: |
| `medium` | 917 |
| `low` | 64 |
| `high` | 12 |

## Unknown spec fields

| key | count |
| --- | ---: |
| `params` | 981 |
| `planned_steps` | 981 |
| `seed` | 685 |
| `git_sha` | 22 |
| `phase` | 17 |
| `job_id` | 9 |

## Inferred terminal status

| key | count |
| --- | ---: |
| `completed` | 568 |
| `crashed` | 198 |
| `<none: unsealed>` | 135 |
| `cancelled` | 86 |
| `timeout` | 6 |

## Anomaly notices

| key | count |
| --- | ---: |
| `schema_drift_missing_keys` | 192 |
| `stranded_non_terminal` | 70 |
| `path_only_identity` | 9 |
| `disputed_terminal_status` | 3 |
| `non_lifecycle_status` | 3 |
| `near_duplicate_keys` | 2 |
| `qualified_terminal_status` | 1 |

