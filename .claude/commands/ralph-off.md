---
description: Disable the Ralph automation loop and prevent automatic pipeline advancement.
allowed-tools: Bash, Read
---
# /ralph-off

Disable Ralph's automatic loop for manual control of project pipeline advancement.

## Usage

```bash
/ralph-off
```

## What It Does

1. **Removes** `.claude/ralph/enabled` flag file
2. **Disables** the Stop hook automation loop
3. **Prevents** automatic dispatch when stopping Claude Code

## When to Use

- Debugging specific projects manually
- Testing individual phases step-by-step
- Preventing accidental automation during sensitive operations
- Manual control and review between each dispatch cycle

## Output

```
✓ Ralph loop disabled
Removed: .claude/ralph/enabled
Status: Manual dispatch only
```

## Related Skills

- **`/ralph-on`**: Enable the Ralph loop
- **`/dispatch`**: Manual dispatch (required when Ralph disabled)
- **`/check-status`**: Monitor project status
