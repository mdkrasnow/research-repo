# Claude Code Hooks - Phase 1 Implementation

This document describes the hooks implemented for the research repository to enable autonomous, parallel, and accurate project management.

## Overview

Hooks are automated actions that execute at specific points in Claude Code's lifecycle. Unlike relying on the LLM to "decide" to run something, hooks **guarantee execution** at the right time, enabling deterministic automation.

## Implemented Hooks

### 1. SessionStart Hook

**File**: `.claude/hooks/session_start.py` / `session_start.sh`

**Purpose**: Injects research context and project status at session initialization.

**Triggers**: When Claude Code starts or resumes a session

**Functionality**:
- Displays current git branch and recent commits
- Categorizes all projects by status:
  - **⭐ Actionable**: Projects with defined `next_action` ready to execute
  - **⏳ Waiting for SLURM**: Projects waiting for job completion (shows time until next poll)
  - **🚫 Blocked**: Projects requiring user input
  - **⊘ Idle**: Projects with no defined next action
- Provides guidance on next steps (suggesting `/dispatch`)

**Output Example**:
```
============================================================
RESEARCH PROJECT CONTEXT BRIEFING
============================================================

📍 Branch: claude/research-code-hooks-vBSt8

📝 Recent commits:
bc4c419 Merge pull request #3 from mdkrasnow/claude/implement-dispatch-848Sy
3e20ef0 Add /dispatch-results skill for tracking dispatch outcomes
...

⭐ ACTIONABLE PROJECTS (2):
  • project-a         [IMPLEMENT] → Set up experiment runner
  • project-b         [DEBUG   ] → Fix SLURM exit code 120

⏳ WAITING FOR SLURM (1):
  • project-c         (check in 15m) [WAIT_SLURM]

🚫 BLOCKED - USER INPUT NEEDED (1):
  • project-d
    → Cluster SSH access not configured. Run scripts/cluster/ssh_b
```

**Benefits**:
- ✅ Eliminates need for manual status briefing
- ✅ Claude starts with full situation awareness
- ✅ Faster decision-making on priority
- ✅ Prevents "what was I working on?" confusion

---

### 2. PermissionRequest Hook

**File**: `.claude/hooks/permission_request.py` / `permission_request.sh`

**Purpose**: Auto-approves safe operations, blocks dangerous ones.

**Triggers**: When Claude Code requests permission for file operations, git commands, etc.

**Auto-Approve Rules**:
- ✅ Read operations (any file)
- ✅ Write/Edit within `projects/<slug>/...` (isolated project files)
- ✅ Git: fetch, pull, add, commit on any branch
- ✅ Git: branch operations

**Auto-Deny Rules**:
- ❌ Force push to any branch
- ❌ Push to `main` or `master` (prevents accidental pushes)
- ❌ Write/Edit to `.claude/ralph/` (Ralph loop config)
- ❌ Write/Edit to `.state/` files directly (use pipeline operations instead)
- ❌ Write/Edit to `.claude/hooks/` scripts
- ❌ Write/Edit to `.claude/settings.json`

**Benefits**:
- ✅ Eliminates permission dialog interruptions for routine operations
- ✅ Prevents dangerous operations automatically
- ✅ Speeds up autonomous operation significantly
- ✅ Maintains safety by protecting critical configuration

**Examples**:
```bash
# Allowed (safe area)
Write to projects/my-project/runs/exp-001/results.md ✅

# Blocked (protected configuration)
Write to projects/my-project/.state/pipeline.json ❌
Write to .claude/ralph/enabled ❌
```

---

### 3. PostToolUse Hook (Code Quality Validation)

**File**: `.claude/hooks/post_tool_use.py` / `post_tool_use.sh`

**Purpose**: Validates code quality after Write/Edit operations.

**Triggers**: After successful Write or Edit tool execution

**Checks Performed**:

**TypeScript/JavaScript Files**:
- ✅ Type checking via `tsc --noEmit` (if available)
- ✅ Detection of `console.log` in non-test files (likely debug code)
- ✅ Detection of TODO/FIXME comments left behind

**Python Files**:
- ✅ Syntax validation via `py_compile`
- ✅ Detection of TODO/FIXME comments

**All Files**:
- ✅ Detection of critical deleted files (package.json, requirements.txt, etc.)

**Output Example**:
```
Code Quality Feedback:
  TypeScript errors in api.ts:
api.ts(45,12): error TS2532: Object is possibly 'undefined'
  ⚠️  api.ts contains console.log (likely debug code)
  📝 api.ts has TODO/FIXME comments:
      Line 89: // TODO: add error handling for edge case
```

**Behavior**:
- ℹ️ **Non-blocking**: Exits with code 0 (always allows continuation)
- ℹ️ **Informative**: Emits warnings to stderr for Claude to see
- ℹ️ **Iterative**: Claude can address feedback in next iteration

**Benefits**:
- ✅ Catches type errors and syntax issues immediately
- ✅ Prevents bad code from reaching SLURM runs
- ✅ Enforces code quality without blocking
- ✅ Especially valuable in parallel `/parallel-implement` mode
- ✅ Reduces manual code review burden

---

## Configuration

All hooks are registered in `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [...],
    "PermissionRequest": [...],
    "PostToolUse": [
      {"matcher": "Write", "hooks": [...]},
      {"matcher": "Edit", "hooks": [...]}
    ],
    "Stop": [...]
  }
}
```

## Usage

These hooks are **automatically active** once configured. They do not require manual activation but can be controlled via:

- **SessionStart**: Runs automatically at session start (always active, informational only)
- **PermissionRequest**: Auto-approves/denies as configured (respects user's final decision if unclear)
- **PostToolUse**: Runs after code changes (informational only, non-blocking)
- **Stop**: Existing Ralph loop (respects configured max iterations)

## Testing

All hooks have been tested for:
- ✅ Correct exit codes (0 for allow, 1 for deny)
- ✅ Proper error detection (TypeScript, Python, debug code)
- ✅ Safe path detection (distinguishes projects/ from .state/)
- ✅ Git operation validation
- ✅ Graceful handling of missing tools

## Future Enhancements

Potential Phase 2 hooks:

1. **SubagentStop Hook**: Validate Task (subagent) completion, auto-retry on failure
2. **UserPromptSubmit Hook**: Enforce research protocol compliance, prevent pipeline corruption
3. **SessionEnd Hook**: Automated archival and session reporting

See parent analysis document for full brainstorm.

## Troubleshooting

### SessionStart Hook not showing output
- Check if session is starting fresh (should see output on stderr)
- Verify `.claude/hooks/session_start.sh` is executable: `ls -la .claude/hooks/`

### PermissionRequest not blocking dangerous operations
- Check `.claude/settings.json` is loading (run `/hooks` command)
- Verify deny rules match your intent
- Note: If not recognized, falls through to user decision (safer default)

### PostToolUse not detecting errors
- Ensure tool (tsc, python3) is installed and in PATH
- Check file_path is correct
- Timeouts (10s default) will not block; check stderr for "[Check timed out]"

## Security Considerations

- Hooks run with `CLAUDE_PROJECT_DIR` environment variable
- All file paths are validated relative to project root
- Critical files (.state/, .claude/ralph/) are protected
- Hooks cannot prevent `Stop` hook from respecting max iterations limit

