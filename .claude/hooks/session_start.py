#!/usr/bin/env python3
"""
SessionStart Hook: Injects research context and project status at session initialization.

Outputs a comprehensive briefing including:
- Active projects and their current phases
- Projects needing attention (actionable work)
- Projects blocked (waiting for SLURM or user input)
- Recent debugging issues
- Git branch and recent commits
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd()))
PROJECTS_DIR = PROJECT_DIR / "projects"


def utc_now():
    return datetime.now(timezone.utc)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def get_git_status():
    """Get current branch and recent commits."""
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()

        recent = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()

        return {"branch": branch, "recent_commits": recent}
    except Exception:
        return {"branch": "unknown", "recent_commits": ""}


def analyze_projects():
    """Analyze all projects and categorize by status."""
    actionable = []
    waiting = []
    blocked = []
    idle = []

    if not PROJECTS_DIR.exists():
        return actionable, waiting, blocked, idle

    now_ts = utc_now().timestamp()

    for pdir in PROJECTS_DIR.iterdir():
        if not pdir.is_dir() or pdir.name.startswith("."):
            continue

        pipe_file = pdir / ".state" / "pipeline.json"
        if not pipe_file.exists():
            continue

        pdata = load_json(pipe_file, {})
        slug = pdir.name
        phase = pdata.get("phase", "UNKNOWN")
        next_action = pdata.get("next_action", "")

        project_info = {
            "slug": slug,
            "phase": phase,
            "next_action": next_action,
        }

        # Check if blocked by user input
        if pdata.get("needs_user_input", {}).get("value", False):
            prompt = pdata.get("needs_user_input", {}).get("prompt", "")
            project_info["user_prompt"] = prompt
            blocked.append(project_info)
            continue

        # Check if no next action (idle)
        if not next_action:
            idle.append(project_info)
            continue

        # Check SLURM waiting
        if phase == "WAIT_SLURM":
            nxt = (pdata.get("slurm_wait") or {}).get("next_poll_after")
            if nxt:
                try:
                    nxt_ts = datetime.fromisoformat(
                        nxt.replace("Z", "+00:00")
                    ).timestamp()
                    if now_ts < nxt_ts:
                        wait_secs = int(nxt_ts - now_ts)
                        project_info["wait_seconds"] = wait_secs
                        waiting.append(project_info)
                        continue
                except Exception:
                    pass

        # Otherwise actionable
        actionable.append(project_info)

    return actionable, waiting, blocked, idle


def format_context():
    """Format context briefing as plain text."""
    git_info = get_git_status()
    actionable, waiting, blocked, idle = analyze_projects()

    lines = [
        "=" * 60,
        "RESEARCH PROJECT CONTEXT BRIEFING",
        "=" * 60,
        "",
        f"📍 Branch: {git_info['branch']}",
        "",
    ]

    if git_info["recent_commits"]:
        lines.extend(
            [
                "📝 Recent commits:",
                git_info["recent_commits"],
                "",
            ]
        )

    if actionable:
        lines.append(f"⭐ ACTIONABLE PROJECTS ({len(actionable)}):")
        for p in actionable:
            lines.append(
                f"  • {p['slug']:20} [{p['phase']:12}] → {p['next_action'][:40]}"
            )
        lines.append("")

    if waiting:
        lines.append(f"⏳ WAITING FOR SLURM ({len(waiting)}):")
        for p in waiting:
            wait_secs = p.get("wait_seconds", 0)
            wait_mins = wait_secs // 60
            lines.append(
                f"  • {p['slug']:20} (check in {wait_mins}m) [{p['phase']}]"
            )
        lines.append("")

    if blocked:
        lines.append(f"🚫 BLOCKED - USER INPUT NEEDED ({len(blocked)}):")
        for p in blocked:
            lines.append(f"  • {p['slug']:20}")
            if p.get("user_prompt"):
                lines.append(f"    → {p['user_prompt'][:60]}")
        lines.append("")

    if idle:
        lines.append(f"⊘ IDLE - No next action defined ({len(idle)}):")
        for p in idle:
            lines.append(f"  • {p['slug']:20} [{p['phase']}]")
        lines.append("")

    lines.extend(
        [
            "💡 SUGGESTED NEXT STEPS:",
            "  1. Review actionable projects above",
            "  2. Run `/dispatch` to advance the next project",
            "  3. Check blocked projects if user input is available",
            "  4. Run `/status` to see full project details",
            "",
            "=" * 60,
        ]
    )

    return "\n".join(lines)


def main():
    context = format_context()
    print(context, file=sys.stderr)


if __name__ == "__main__":
    main()
