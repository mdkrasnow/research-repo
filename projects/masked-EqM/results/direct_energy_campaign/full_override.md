# User-directed real-scale screen

## Decision

The Stage 10 pilot decision was negative: native scalar energy was technically
valid but did not show a pilot-scale advantage.  On 2026-07-23, the user
explicitly directed that the matched comparison proceed anyway.  This document
records that override; it does not reinterpret the pilot result as a pass.

## Scope

The submitted screen is the project's established real ImageNet-1K protocol:
EqM-B/2, 256px, one epoch (40,000 updates), matched seed, optimizer, batch
size, data order, EMA, and checkpoint policy across `none`, `dot`, and
`direct`.  Seeds 0--2 will be run after each preceding seed passes the
startup/memory preflight.  This is a real-scale one-epoch screen, not an
80-epoch final-quality claim.

## Seed 0 jobs

| Arm | Job | State at submission | Commit |
| --- | ---: | --- | --- |
| `none` | 34721182 | PENDING | `858bb0f` |
| `dot` | 34721183 | PENDING | `858bb0f` |
| `direct` | 34721184 | PENDING | `858bb0f` |

Scheduler events are recorded by job 34721386 in `monitor/`.  The prior
monitor job 34721354 was cancelled after it wrote malformed JSONL; its output
was preserved as `monitor/scheduler_events_malformed_34721354.jsonl` and the
replacement was regression-checked for valid JSONL.
