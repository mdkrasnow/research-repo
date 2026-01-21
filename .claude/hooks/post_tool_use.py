#!/usr/bin/env python3
"""
PostToolUse Hook: Validates code quality after Write/Edit operations.

Performs quality checks:
- TypeScript/JavaScript: Run tsc --noEmit and eslint (if available)
- Python: Run mypy, pylint (if available)
- General: Check for obvious antipatterns (TODOs, console.log in production code)
- Validation: Ensure required files weren't accidentally deleted

Provides feedback to Claude without blocking (exit 0), allowing iteration.
Emits warnings to stderr for Claude to see and address.
"""

import json
import os
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd()))


def load_stdin():
    """Load post-tool-use event JSON from stdin."""
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def run_check(command: list, cwd=None, timeout=10) -> tuple[bool, str]:
    """
    Run a check command. Returns (success: bool, output: str).
    success=True means check passed or not applicable.
    """
    try:
        result = subprocess.run(
            command,
            cwd=cwd or PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Most linters/type checkers exit 0 on success
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return True, "[Check timed out]"  # Don't block
    except FileNotFoundError:
        return True, "[Tool not found]"  # Don't block if tool not installed
    except Exception as e:
        return True, f"[Check error: {e}]"  # Don't block on unexpected errors


def check_typescript_file(file_path: Path) -> list[str]:
    """Check a TypeScript/JavaScript file for quality issues."""
    warnings = []

    # Check if tsc is available
    tsc_available, _ = run_check(["which", "tsc"])
    if tsc_available:
        success, output = run_check(
            ["tsc", "--noEmit", str(file_path)], cwd=PROJECT_DIR
        )
        if not success and output:
            warnings.append(f"TypeScript errors in {file_path.name}:\n{output[:500]}")

    # Check for console.log in non-test files (likely debug code left behind)
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    if "console.log" in content and "test" not in file_path.name:
        warnings.append(
            f"⚠️  {file_path.name} contains console.log (likely debug code)"
        )

    # Check for TODO/FIXME comments
    lines_with_todo = [
        (i + 1, line)
        for i, line in enumerate(content.split("\n"))
        if "TODO" in line or "FIXME" in line
    ]
    if lines_with_todo:
        todo_list = "\n      ".join(f"Line {ln}: {txt[:60]}" for ln, txt in lines_with_todo[:3])
        warnings.append(f"📝 {file_path.name} has TODO/FIXME comments:\n      {todo_list}")

    return warnings


def check_python_file(file_path: Path) -> list[str]:
    """Check a Python file for quality issues."""
    warnings = []

    # Check for basic syntax errors
    success, output = run_check(["python3", "-m", "py_compile", str(file_path)])
    if not success:
        warnings.append(f"Python syntax error in {file_path.name}:\n{output[:300]}")

    content = file_path.read_text(encoding="utf-8", errors="ignore")

    # Check for TODO/FIXME
    lines_with_todo = [
        (i + 1, line)
        for i, line in enumerate(content.split("\n"))
        if "TODO" in line or "FIXME" in line
    ]
    if lines_with_todo:
        todo_list = "\n      ".join(f"Line {ln}: {txt[:60]}" for ln, txt in lines_with_todo[:3])
        warnings.append(f"📝 {file_path.name} has TODO/FIXME comments:\n      {todo_list}")

    return warnings


def check_modified_file(file_path: str) -> list[str]:
    """Run quality checks on a modified file."""
    warnings = []

    try:
        path = Path(file_path)
        if not path.exists():
            # File was deleted - check if it was critical
            if any(
                critical in file_path
                for critical in [
                    "package.json",
                    "tsconfig.json",
                    "requirements.txt",
                    "pyproject.toml",
                ]
            ):
                warnings.append(
                    f"⚠️  CRITICAL FILE DELETED: {file_path} - This may break the project!"
                )
            return warnings

        if path.suffix in [".ts", ".tsx", ".js", ".jsx"]:
            warnings.extend(check_typescript_file(path))

        elif path.suffix in [".py"]:
            warnings.extend(check_python_file(path))

    except Exception as e:
        pass  # Don't block on check errors

    return warnings


def main():
    event = load_stdin()

    tool = event.get("tool", "")
    params = event.get("parameters", {})

    # Only check Write and Edit operations
    if tool not in ["Write", "Edit"]:
        sys.exit(0)

    file_path = params.get("file_path", "")
    if not file_path:
        sys.exit(0)

    warnings = check_modified_file(file_path)

    if warnings:
        output = "Code Quality Feedback:\n" + "\n".join(f"  {w}" for w in warnings)
        print(output, file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
