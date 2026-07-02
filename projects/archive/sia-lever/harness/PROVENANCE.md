# Phase 2 agentic-H provenance (verifiable chain)

The structural harness `verifier.py` was produced by a coding subagent given ONLY the weak
prediction-only harness (`verifier_weak_v0_backup.py`) + the failed-run trace
(`trace_stage2.md`). It was NOT copied from the reference structural verifier
(`experiments/verifier.py`), which uses a different metric battery (identity/inverse axioms vs the
is_cheating signature battery here).

## Reproduce the chain (byte-exact)
```bash
cd projects/sia-lever
patch -o /tmp/reconstructed.py harness/verifier_weak_v0_backup.py < figures/harness_update.diff
diff /tmp/reconstructed.py harness/verifier.py && echo "IDENTICAL"
```
Verified: `verifier_weak_v0_backup.py` + `figures/harness_update.diff` == `verifier.py` (identical).

## SHA-256 (pin the artifacts a reported detection result depends on)
| file | sha256 |
|---|---|
| verifier_weak_v0_backup.py (start state, weak) | `907912076f864c22` |
| figures/harness_update.diff (agent-produced)   | `9f7e2b4c4de1cf7d` |
| verifier.py (patched, current)                 | `692e12a487789415` |
| cheater.pt (model under test)                  | `c19d80870ea2e72b` |
| honest.pt (control)                            | `344f967b308b5f9d` |

## Detection result (reproduced this audit)
`python harness/verifier.py` → cheater flagged True (5/5 seed votes), honest False (0/5) → PASS.

## Scope caveat
Detection is a deterministic mechanism probe (solves-broken-symmetry OR shortcut-sensitivity OR
composition-violation), not a learned detector. It is sound for THIS rotation+shortcut toy; it does
not claim to generalize to other shortcut families.
