#!/usr/bin/env python3
"""
SessionEnd Hook: Generates session summary, archives results, and logs metrics.

Triggered when Claude Code session ends. Performs:
- Snapshot of project state at end of session
- Records completed actions during this session
- Generates session report with metrics
- Archives run artifacts if requested
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd()))
PROJECTS_DIR = PROJECT_DIR / "projects"
SESSIONS_DIR = PROJECT_DIR / ".claude" / "sessions"


def utc_now():
    return datetime.now(timezone.utc)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def get_git_info():
    """Get git commit info at session end."""
    try:
        import subprocess

        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()

        return {"sha": sha[:8], "branch": branch}
    except Exception:
        return {"sha": "unknown", "branch": "unknown"}


def collect_project_changes():
    """Analyze changes to each project's pipeline state."""
    changes = defaultdict(lambda: {"phase_changed": False, "actions_completed": 0})

    if not PROJECTS_DIR.exists():
        return changes

    for pdir in PROJECTS_DIR.iterdir():
        if not pdir.is_dir() or pdir.name.startswith("."):
            continue

        pipe_file = pdir / ".state" / "pipeline.json"
        if not pipe_file.exists():
            continue

        pdata = load_json(pipe_file, {})
        slug = pdir.name

        # Check for completed runs
        completed = pdata.get("completed_runs", [])
        if completed:
            changes[slug]["actions_completed"] = len(completed)

        # Check phase
        phase = pdata.get("phase", "UNKNOWN")
        changes[slug]["current_phase"] = phase

    return dict(changes)


def count_new_artifacts():
    """Count new artifacts created during this session (recent runs)."""
    artifact_count = 0
    recent_run_count = 0
    now = utc_now()
    one_hour_ago = (now.timestamp() - 3600)

    if not PROJECTS_DIR.exists():
        return artifact_count, recent_run_count

    for pdir in PROJECTS_DIR.iterdir():
        if not pdir.is_dir() or pdir.name.startswith("."):
            continue

        runs_dir = pdir / "runs"
        if not runs_dir.exists():
            continue

        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check if run is recent (within last hour)
            try:
                mod_time = run_dir.stat().st_mtime
                if mod_time > one_hour_ago:
                    recent_run_count += 1
                    # Count files in this run
                    for file in run_dir.rglob("*"):
                        if file.is_file():
                            artifact_count += 1
            except Exception:
                pass

    return artifact_count, recent_run_count


def generate_session_report():
    """Generate comprehensive session report."""
    session_end_time = utc_now()
    git_info = get_git_info()
    project_changes = collect_project_changes()
    artifact_count, recent_run_count = count_new_artifacts()

    # Check for any projects that moved to completed
    completed_projects = [
        slug for slug, changes in project_changes.items()
        if changes.get("current_phase") == "COMPLETED"
    ]

    # Check for any projects that moved to DEBUG
    debug_projects = [
        slug for slug, changes in project_changes.items()
        if changes.get("current_phase") == "DEBUG"
    ]

    report = {
        "timestamp": session_end_time.isoformat().replace("+00:00", "Z"),
        "git": git_info,
        "metrics": {
            "projects_touched": len(project_changes),
            "projects_completed": len(completed_projects),
            "projects_in_debug": len(debug_projects),
            "artifacts_created": artifact_count,
            "recent_runs": recent_run_count,
        },
        "completed_projects": completed_projects,
        "debug_projects": debug_projects,
        "project_changes": project_changes,
    }

    return report


def save_session_report(report: dict):
    """Save session report to sessions directory."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Use timestamp as filename
    timestamp = report["timestamp"].replace(":", "-").replace("Z", "")
    session_file = SESSIONS_DIR / f"session-{timestamp}.json"

    save_json(session_file, report)

    # Also append to a running log
    log_file = SESSIONS_DIR / "session-log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")

    return session_file


def format_session_summary(report: dict) -> str:
    """Format report as human-readable summary."""
    lines = [
        "",
        "=" * 60,
        "SESSION SUMMARY",
        "=" * 60,
        "",
        f"📅 Session ended: {report['timestamp']}",
        f"🌿 Branch: {report['git']['branch']} ({report['git']['sha']})",
        "",
        "📊 METRICS:",
        f"  • Projects touched: {report['metrics']['projects_touched']}",
        f"  • Artifacts created: {report['metrics']['artifacts_created']} files in {report['metrics']['recent_runs']} runs",
        f"  • Projects completed: {report['metrics']['projects_completed']}",
        f"  • Projects in debug: {report['metrics']['projects_in_debug']}",
        "",
    ]

    if report["completed_projects"]:
        lines.append("✅ COMPLETED PROJECTS:")
        for proj in report["completed_projects"]:
            lines.append(f"   • {proj}")
        lines.append("")

    if report["debug_projects"]:
        lines.append("🔧 PROJECTS MOVED TO DEBUG:")
        for proj in report["debug_projects"]:
            lines.append(f"   • {proj}")
        lines.append("")

    if report["project_changes"]:
        lines.append("📝 PROJECT CHANGES:")
        for slug, changes in sorted(report["project_changes"].items()):
            phase = changes.get("current_phase", "?")
            actions = changes.get("actions_completed", 0)
            if actions > 0:
                lines.append(f"   • {slug:20} [{phase:12}] completed {actions} action(s)")
        lines.append("")

    lines.extend([
        "💾 Session report saved to: .claude/sessions/",
        "📋 View session history: cat .claude/sessions/session-log.jsonl",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


def main():
    report = generate_session_report()
    save_session_report(report)
    summary = format_session_summary(report)
    print(summary, file=sys.stderr)


if __name__ == "__main__":
    main()
