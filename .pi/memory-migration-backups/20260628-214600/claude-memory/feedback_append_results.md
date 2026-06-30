---
name: Append to results.tsv, never overwrite
description: Always append new round results to results.tsv, never drop previous entries
type: feedback
---

ALWAYS append new results to results.tsv. Never overwrite or remove previous entries (including SKIP/CRASH rows).

**Why:** User caught that I was rewriting results.tsv each round, dropping SKIP entries from earlier rounds. The full history is the experiment record.

**How to apply:** After a tournament, append only the new round's rows. Never use Write to replace the whole file — use Edit to add lines at the end, or read first and append.
