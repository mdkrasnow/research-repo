# Claude Code Hooks - Phase 1 & Phase 2 Implementation

This document describes the hooks implemented for the research repository to enable autonomous, parallel, and accurate project management.

**Phase 1**: SessionStart, PostToolUse, Stop (existing)
**Phase 2**: SubagentStop, SessionEnd

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

### 2. PostToolUse Hook (Code Quality Validation)

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

### 3. SubagentStop Hook (Phase 2)

**File**: `.claude/hooks/subagent_stop.py` / `subagent_stop.sh`

**Purpose**: Validates Task (subagent) completion and manages retries.

**Triggers**: When a Task tool completes execution

**Functionality**:
- ✅ Checks if subagent produced `agent_id` (successful execution indicator)
- ✅ Analyzes output for error keywords and transient failures
- ✅ Detects incomplete executions (no output, no agent_id)
- ✅ Identifies retry-able errors (connection timeout, temporary unavailable)
- ✅ Flags non-transient errors for manual review
- ✅ Logs all subagent results to audit trail

**Output Example**:
```
Successful:
  ✅ Subagent completed successfully (agent_id: a1b2c3d4e5f6)

With Errors (Transient - Retry):
  🔍 Subagent Completion Analysis:
    ⚠️  Errors detected: connection refused, timeout
    🔄 Suggests transient failure - consider retry

With Errors (Non-Transient - Review):
  🔍 Subagent Completion Analysis:
    ⚠️  No agent_id returned - execution may have failed
    ⚠️  Errors detected: syntax error, module not found
    ℹ️  Non-transient errors detected; requires review
```

**Behavior**:
- ℹ️ **Non-blocking**: Exit code 0 (always allows continuation)
- ℹ️ **Informative**: Emits detailed analysis to stderr
- ℹ️ **Logged**: All results saved to `.claude/subagent-logs/`

**Benefits**:
- ✅ Catches silent failures in parallel Task execution
- ✅ Distinguishes transient vs. permanent failures
- ✅ Provides clear feedback for human/LLM decision on retries
- ✅ Enables safe parallel `@parallel-implement` mode
- ✅ Creates audit trail for debugging failed tasks
- ✅ Reduces need for manual investigation

---

### 4. SessionEnd Hook (Phase 2)

**File**: `.claude/hooks/session_end.py` / `session_end.sh`

**Purpose**: Generates session summary, archives results, and logs metrics.

**Triggers**: When Claude Code session ends

**Functionality**:
- ✅ Captures final project state (phase, completed actions)
- ✅ Records metrics: artifacts created, runs executed, projects completed/debugged
- ✅ Generates human-readable summary with emoji status indicators
- ✅ Saves structured JSON report to `.claude/sessions/`
- ✅ Appends to session log for historical analysis
- ✅ Includes git commit SHA and branch for reproducibility

**Output Example**:
```
============================================================
SESSION SUMMARY
============================================================

📅 Session ended: 2026-01-21T05:47:09Z
🌿 Branch: claude/feature-branch (f5fee516)

📊 METRICS:
  • Projects touched: 2
  • Artifacts created: 11 files in 11 runs
  • Projects completed: 0
  • Projects in debug: 1

✅ COMPLETED PROJECTS:
   • experiment-001

🔧 PROJECTS MOVED TO DEBUG:
   • experiment-002

📝 PROJECT CHANGES:
   • experiment-001  [COMPLETED] completed 5 action(s)
   • experiment-002  [DEBUG    ] completed 3 action(s)

💾 Session report saved to: .claude/sessions/
📋 View session history: cat .claude/sessions/session-log.jsonl
```

**Behavior**:
- ℹ️ **Non-blocking**: Exit code 0 (informational only)
- ℹ️ **Persistent**: Saves to `.claude/sessions/session-*.json` + `.claude/sessions/session-log.jsonl`
- ℹ️ **Queryable**: JSONL log enables analysis of session history

**Benefits**:
- ✅ Audit trail for all work performed
- ✅ Enables analysis of project progression over time
- ✅ Clear baseline for resuming work in next session
- ✅ Metrics for understanding workflow efficiency
- ✅ Reproducibility: git SHA captured for exact replay
- ✅ Accountability: who did what and when

---

## Configuration

All hooks are registered in `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [...],
    "PostToolUse": [
      {"matcher": "Write", "hooks": [...]},
      {"matcher": "Edit", "hooks": [...]}
    ],
    "SubagentStop": [...],
    "SessionEnd": [...],
    "Stop": [...]
  }
}
```

## Usage

These hooks are **automatically active** once configured. They do not require manual activation but can be controlled via:

- **SessionStart**: Runs automatically at session start (always active, informational only)
- **PostToolUse**: Runs after code changes (informational only, non-blocking)
- **SubagentStop**: Runs when Task completes (informational only, non-blocking, audited)
- **SessionEnd**: Runs when session ends (informational only, generates reports)
- **Stop**: Existing Ralph loop (respects configured max iterations)

## Testing

All hooks have been tested for:

**Phase 1 Hooks**:
- ✅ SessionStart: Correctly categorizes all projects by status
- ✅ PostToolUse: Detects TypeScript/Python errors, console.log, TODOs, critical file deletions

**Phase 2 Hooks**:
- ✅ SubagentStop: Detects successful completion (agent_id), error conditions, transient failures
- ✅ SessionEnd: Generates accurate session summaries, logs to persistent storage

## Future Enhancements

Potential Phase 3 hooks:

1. **UserPromptSubmit Hook**: Enforce research protocol compliance, prevent pipeline corruption

See parent analysis document for full brainstorm.

## Troubleshooting

### SessionStart Hook not showing output
- Check if session is starting fresh (should see output on stderr)
- Verify `.claude/hooks/session_start.sh` is executable: `ls -la .claude/hooks/`

### PostToolUse not detecting errors
- Ensure tool (tsc, python3) is installed and in PATH
- Check file_path is correct
- Timeouts (10s default) will not block; check stderr for "[Check timed out]"

### SubagentStop not detecting failures
- Check stderr for analysis output (should appear when Task completes)
- Verify `.claude/hooks/subagent_stop.sh` is executable
- Check `.claude/subagent-logs/` for detailed audit records
- Note: Always returns exit 0 (non-blocking) - look at stderr feedback for status

### SessionEnd not generating reports
- Verify `.claude/hooks/session_end.sh` is executable
- Check `.claude/sessions/` directory exists and is writable
- Ensure you have projects in `projects/` with `.state/pipeline.json` files
- View session history: `cat .claude/sessions/session-log.jsonl | jq`

## Security Considerations

- Hooks run with `CLAUDE_PROJECT_DIR` environment variable
- All file paths are validated relative to project root
- Critical files (.state/, .claude/ralph/) are protected
- Hooks cannot prevent `Stop` hook from respecting max iterations limit

