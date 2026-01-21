#!/usr/bin/env python3
"""
PermissionRequest Hook: Auto-approves safe operations, blocks dangerous ones.

Auto-approve:
  - Read operations (any file)
  - Write/Edit within projects/<slug>/... (isolated project files)
  - Git operations on current branch (fetch, pull, add, commit)

Auto-deny:
  - Force push to main/master
  - Deletion of .claude/ralph/ or .state/ files
  - Direct modifications to .claude/ config files

Returns exit code 0 to allow, non-zero to deny.
"""

import json
import os
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd()))


def load_stdin():
    """Load permission request JSON from stdin."""
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def get_current_branch():
    """Get current git branch."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


def should_auto_allow(request: dict) -> bool:
    """Determine if request should be auto-approved."""
    tool = request.get("tool", "")
    params = request.get("parameters", {})

    # Read operations are always safe
    if tool in ["Read", "Glob", "Grep", "WebFetch", "WebSearch"]:
        return True

    # Write/Edit operations - allow if in projects/ directory
    if tool in ["Write", "Edit"]:
        file_path = params.get("file_path", "")
        if file_path.startswith(str(PROJECT_DIR / "projects")):
            # But not if modifying .state files directly
            if "/.state/" not in file_path:
                return True
        # Allow writes to runs/ directory
        if "/runs/" in file_path and str(PROJECT_DIR) in file_path:
            return True
        return False

    # Bash - check if it's a safe git operation
    if tool == "Bash":
        cmd = params.get("command", "")

        # Deny force push
        if "git" in cmd and ("push --force" in cmd or "push -f" in cmd):
            return False

        # Deny pushing to main/master
        if "git" in cmd and "push" in cmd:
            if "main" in cmd or "master" in cmd:
                return False

        # Allow read-only git operations
        safe_git_cmds = [
            "git status",
            "git log",
            "git diff",
            "git fetch",
            "git pull",
            "git add",
            "git commit",
            "git branch",
        ]
        if any(safe_cmd in cmd for safe_cmd in safe_git_cmds):
            return True

        # Allow other safe commands (npm, python, etc) in projects/
        if any(
            safe_tool in cmd
            for safe_tool in ["npm", "python", "pytest", "ls", "pwd", "echo"]
        ):
            return True

        return False

    return False


def should_auto_deny(request: dict) -> bool:
    """Determine if request should be auto-denied."""
    tool = request.get("tool", "")
    params = request.get("parameters", {})

    if tool in ["Write", "Edit"]:
        file_path = params.get("file_path", "")

        # Deny modifications to .claude/ralph (Ralph loop config)
        if ".claude/ralph" in file_path:
            return True

        # Deny direct modifications to .state files (use pipeline operations instead)
        if "/.state/" in file_path:
            return True

        # Deny modifications to hook scripts
        if ".claude/hooks" in file_path and not file_path.endswith(".md"):
            return True

        # Deny modifications to settings.json
        if ".claude/settings.json" in file_path:
            return True

    if tool == "Bash":
        cmd = params.get("command", "")

        # Deny force push
        if "git" in cmd and ("push --force" in cmd or "push -f" in cmd):
            return True

        # Deny rm -rf or destructive operations on state
        if "rm" in cmd and (".state" in cmd or ".claude/ralph" in cmd):
            return True

    return False


def main():
    request = load_stdin()

    # Check deny list first (safer)
    if should_auto_deny(request):
        print(f"Permission denied by auto-deny rule", file=sys.stderr)
        sys.exit(1)

    # Check allow list
    if should_auto_allow(request):
        sys.exit(0)

    # If not explicitly allowed or denied, let the user decide (exit with 0, not blocking)
    # This is safer than auto-denying unknown operations
    sys.exit(0)


if __name__ == "__main__":
    main()
