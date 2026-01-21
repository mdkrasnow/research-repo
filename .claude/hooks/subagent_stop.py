#!/usr/bin/env python3
"""
SubagentStop Hook: Validates Task (subagent) completion and handles retries.

Triggered when a Task tool completes. Evaluates:
- Did the subagent produce an agent_id (successful execution)?
- Should we retry (transient failure, incomplete output)?
- Did it produce expected artifacts?
- Should we escalate to DEBUG phase if failed?

Returns exit code 0 to continue, non-zero to mark as failed/needs review.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd()))


def load_stdin():
    """Load subagent-stop event JSON from stdin."""
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, obj):
    """Save JSON with atomic write."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def analyze_subagent_result(event: dict) -> dict:
    """
    Analyze the subagent result. Returns:
    {
        "success": bool,
        "has_agent_id": bool,
        "output_lines": int,
        "has_errors": bool,
        "error_keywords": list,
        "should_retry": bool,
        "feedback": str
    }
    """
    analysis = {
        "success": False,
        "has_agent_id": False,
        "output_lines": 0,
        "has_errors": False,
        "error_keywords": [],
        "should_retry": False,
        "feedback": "",
    }

    # Check if we have agent_id (successful execution)
    agent_id = event.get("agent_id", "")
    if agent_id:
        analysis["has_agent_id"] = True
        analysis["success"] = True

    # Analyze output
    output = event.get("output", "")
    if output:
        lines = output.split("\n")
        analysis["output_lines"] = len([l for l in lines if l.strip()])

    # Check for error indicators
    error_keywords = [
        "error:",
        "failed:",
        "exception",
        "traceback",
        "timeout",
        "connection refused",
        "permission denied",
        "no such file",
        "module not found",
        "syntax error",
    ]

    output_lower = output.lower() if output else ""
    for keyword in error_keywords:
        if keyword in output_lower:
            analysis["has_errors"] = True
            analysis["error_keywords"].append(keyword)

    # Determine if retryable
    # Transient errors that might succeed on retry
    transient_errors = [
        "timeout",
        "connection refused",
        "temporarily unavailable",
        "try again",
    ]
    for error in transient_errors:
        if error in output_lower:
            analysis["should_retry"] = True
            break

    # If no agent_id and no output, likely incomplete/failed
    if not agent_id and not output:
        analysis["should_retry"] = True
        analysis["feedback"] = "No output or agent_id; likely incomplete"

    # If agent_id exists but has errors, likely partial failure
    if agent_id and analysis["has_errors"]:
        analysis["feedback"] = f"Completed with errors: {', '.join(set(analysis['error_keywords']))}"

    # If errors but no retry indicators, mark for manual review
    if analysis["has_errors"] and not analysis["should_retry"]:
        analysis["success"] = False
        analysis["feedback"] = "Non-transient errors detected; requires review"

    return analysis


def get_project_for_subagent(event: dict) -> str:
    """Try to determine which project this subagent was working on."""
    # Look for project slug in context or description
    description = event.get("description", "").lower()
    context = event.get("context", "").lower()

    PROJECTS_DIR = PROJECT_DIR / "projects"
    if PROJECTS_DIR.exists():
        for pdir in PROJECTS_DIR.iterdir():
            slug = pdir.name.lower()
            if slug in description or slug in context:
                return pdir.name

    return ""


def check_project_state(project_slug: str) -> dict:
    """Get current state of a project."""
    if not project_slug:
        return {}

    pipe_file = PROJECT_DIR / "projects" / project_slug / ".state" / "pipeline.json"
    return load_json(pipe_file, {})


def log_subagent_result(event: dict, analysis: dict):
    """Log the subagent result to a debug file."""
    logs_dir = PROJECT_DIR / ".claude" / "subagent-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "-").split(".")[0]
    log_file = logs_dir / f"subagent-{timestamp}.json"

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "agent_id": event.get("agent_id", "unknown"),
        "description": event.get("description", ""),
        "analysis": analysis,
        "full_event": event,
    }

    save_json(log_file, log_entry)
    return log_file


def format_feedback(analysis: dict) -> str:
    """Format feedback message for Claude."""
    if analysis["success"]:
        return f"✅ Subagent completed successfully (agent_id: {analysis.get('agent_id', '?')})"

    lines = ["🔍 Subagent Completion Analysis:"]

    if not analysis["has_agent_id"]:
        lines.append("  ⚠️  No agent_id returned - execution may have failed")

    if analysis["output_lines"] == 0:
        lines.append("  ⚠️  No output produced")
    else:
        lines.append(f"  ✓ Output: {analysis['output_lines']} lines")

    if analysis["has_errors"]:
        errors_str = ", ".join(set(analysis["error_keywords"]))
        lines.append(f"  ⚠️  Errors detected: {errors_str}")

    if analysis["should_retry"]:
        lines.append("  🔄 Suggests transient failure - consider retry")

    if analysis["feedback"]:
        lines.append(f"  ℹ️  {analysis['feedback']}")

    return "\n".join(lines)


def main():
    event = load_stdin()

    # Analyze the subagent result
    analysis = analyze_subagent_result(event)

    # Log for audit trail
    log_file = log_subagent_result(event, analysis)

    # Generate feedback
    feedback = format_feedback(analysis)
    print(feedback, file=sys.stderr)

    # Exit code 0 always (non-blocking), but provide clear feedback for Claude to act on
    sys.exit(0)


if __name__ == "__main__":
    main()
