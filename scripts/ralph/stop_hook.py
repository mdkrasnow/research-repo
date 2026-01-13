#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd()))
RALPH_DIR = PROJECT_DIR / ".claude" / "ralph"
ENABLED_FILE = RALPH_DIR / "enabled"
CONFIG_FILE = RALPH_DIR / "config.json"
LOOP_STATE = RALPH_DIR / "loop-state.json"

DEFAULT_CONFIG = {"max_iterations": 25, "batch_size": 3, "min_seconds_between_blocks": 0}

def utc_now():
    return datetime.now(timezone.utc)

def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json_atomic(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

def is_actionable(p: dict, now_ts: float) -> bool:
    if not isinstance(p, dict):
        return False
    if p.get("needs_user_input", {}).get("value", False):
        return False
    if not p.get("next_action"):
        return False
    if p.get("phase") == "WAIT_SLURM":
        nxt = (p.get("slurm_wait") or {}).get("next_poll_after")
        if not nxt:
            return True
        try:
            nxt_ts = datetime.fromisoformat(nxt.replace("Z", "+00:00")).timestamp()
            return now_ts >= nxt_ts
        except Exception:
            return True
    return True

def main():
    if not ENABLED_FILE.exists():
        sys.exit(0)

    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}

    session_id = payload.get("session_id", "unknown")
    stop_hook_active = bool(payload.get("stop_hook_active", False))

    cfg = load_json(CONFIG_FILE, DEFAULT_CONFIG)
    max_iter = int(cfg.get("max_iterations", DEFAULT_CONFIG["max_iterations"]))

    state = load_json(LOOP_STATE, {"session_id": session_id, "iterations_blocked": 0, "last_blocked_at": None})
    if state.get("session_id") != session_id:
        state = {"session_id": session_id, "iterations_blocked": 0, "last_blocked_at": None}

    if int(state.get("iterations_blocked", 0)) >= max_iter:
        sys.exit(0)

    projects_dir = PROJECT_DIR / "projects"
    now_ts = utc_now().timestamp()
    actionable = False
    if projects_dir.exists():
        for pdir in projects_dir.iterdir():
            if not pdir.is_dir() or pdir.name.startswith("."):
                continue
            pipe = pdir / ".state" / "pipeline.json"
            if not pipe.exists():
                continue
            pdata = load_json(pipe, {})
            if is_actionable(pdata, now_ts):
                actionable = True
                break

    if not actionable:
        sys.exit(0)

    state["iterations_blocked"] = int(state.get("iterations_blocked", 0)) + 1
    state["last_blocked_at"] = utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")
    save_json_atomic(LOOP_STATE, state)

    msg = "\n".join([
        "Ralph loop: actionable work exists in this repo.",
        "Next step: run `/dispatch` to advance projects.",
        "To pause the loop: run `/ralph-off` (or delete .claude/ralph/enabled).",
        f"(loop iteration {state['iterations_blocked']}/{max_iter}; stop_hook_active={stop_hook_active})",
    ])
    print(msg, file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
