---
description: Enable the Ralph automation loop for continuous project pipeline advancement.
allowed-tools: Bash, Read
---
# /ralph-on

Enable Ralph's automatic loop to continuously advance projects through their pipeline phases.

## Usage

```bash
/ralph-on
```

## What It Does

1. **Creates** `.claude/ralph/enabled` flag file
2. **Enables** the Stop hook automation loop
3. **Allows** Ralph to autonomously dispatch projects when stopped

## Ralph Loop

When enabled, Ralph will:

- Automatically call `/dispatch` when you try to stop Claude Code
- Advance up to `batch_size` projects by one step each
- Update pipeline documentation (pipeline.json, queue.md, debugging.md)
- Continue until all projects complete or user intervention needed

## Output

```
✓ Ralph loop enabled
Created: .claude/ralph/enabled
Status: Ready to auto-dispatch on stop
```

## Related Skills

- **`/ralph-off`**: Disable the Ralph loop
- **`/dispatch`**: Manual dispatch (Ralph calls this automatically)
- **`/check-status`**: Monitor Ralph's progress
