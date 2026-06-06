#!/usr/bin/env python3
"""Reference target agent for the SIA-Lever custom task.

Reads public traces, queries an OpenAI-compatible model (gpt-oss-120b by default) for a lever
action per trace, parses strict JSON, and writes submission.json into the generation dir.

Env: GPT_OSS_BASE_URL, GPT_OSS_API_KEY|NEBIUS_API_KEY|OPENAI_API_KEY, GPT_OSS_MODEL.
  --gen-dir   where to write submission.json (default: ./)
  --dry-run   skip API; emit PROMOTE placeholders (plumbing test)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# resolve project gpt_oss/ for shared prompt+parser
HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent
PROJ = TASK_DIR.parent
sys.path.insert(0, str(PROJ / "gpt_oss"))
from lever_io import build_messages, parse_action  # noqa: E402

PUBLIC = TASK_DIR / "data" / "public" / "traces_public.jsonl"


def load_public():
    rows = []
    with open(PUBLIC) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=".")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_public()
    if args.limit:
        rows = rows[:args.limit]

    chat = client = None
    if not args.dry_run:
        from client import make_client, chat as _chat
        client, chat = make_client(base_url=args.base_url), _chat

    preds = []
    for r in rows:
        if args.dry_run:
            action, reason = "PROMOTE", "dry-run placeholder"
        else:
            raw = chat(build_messages(r["trace_text"]), model=args.model,
                       base_url=args.base_url, client=client)
            action, reason, _ = parse_action(raw)
            action = action or "KILL"
        preds.append({"episode_id": r["episode_id"], "action": action, "reason": reason})

    gen = Path(args.gen_dir)
    (gen / "results").mkdir(parents=True, exist_ok=True)
    out = gen / "results" / "submission.json"
    out.write_text(json.dumps({"predictions": preds}, indent=2))
    # also write at gen-dir root for evaluators that look there first
    (gen / "submission.json").write_text(json.dumps({"predictions": preds}, indent=2))
    print(f"wrote {len(preds)} predictions -> {out}")


if __name__ == "__main__":
    main()
