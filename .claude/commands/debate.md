---
description: Run a structured debate to resolve project decisions and document outcomes.
allowed-tools: Read, Write, Edit, Task, Bash(mkdir:*)
argument-hint: "<question>" --project <slug>
---
# /debate

Conduct a structured 3-round debate to resolve project decisions with documented reasoning.

## Usage

```bash
/debate "Should we use technique X or Y?" --project my-experiment
/debate "Is the model architecture optimal?" --project baseline
```

## What It Does

1. **Frames** the decision question clearly
2. **Generates** multiple positions (typically 3+)
3. **Conducts** 3 rounds of debate between positions
4. **Evaluates** evidence and trade-offs
5. **Reaches** a consensus decision
6. **Documents** outcome in project documentation
7. **Records** decision in `pipeline.json` events log

## Decision Rounds

| Round | Purpose |
|-------|---------|
| **Opening** | Present initial positions with arguments |
| **Response** | Address counterarguments, refine stance |
| **Closing** | Final synthesis and consensus building |

## Output

Decision documented to:

- **`documentation/debates.md`**: Full debate transcript and reasoning
- **`pipeline.json`**: Decision recorded in events log with timestamp
- **`pipeline.phase`**: Advanced to next phase (if decision unlocks progress)

Example output:
```markdown
# Decision: Technique Selection

## Question
Should we use technique X or Y?

## Debate Summary
### Position A: Use X
- Pros: Faster training, proven baseline
- Cons: Limited expressiveness

### Position B: Use Y
- Pros: Better accuracy potential
- Cons: 3x slower training

## Decision
**Consensus: Hybrid approach** — Use X for baseline, evaluate Y with constrained resources

## Rationale
...
```

## Error Handling

- **Invalid question**: Returns error "Question must be formatted as a clear choice"
- **No debate positions**: Falls back to binary (for) vs (against)
- **Consensus unreachable**: Documents dissent and explores compromise options

## Related Skills

- **`/dispatch`**: Main orchestrator (may call this for DEBATE phase)
- **`/make-project`**: Uses debate for experiment design decisions
- **`/check-status`**: See project decisions in pipeline events
