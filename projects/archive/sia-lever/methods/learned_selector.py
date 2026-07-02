"""Learned lever selector backed by a model (base gpt-oss-120b or gpt-oss-120b + LoRA).

Two ways to get actions:
  - live(): query an OpenAI-compatible endpoint per episode (needs creds/endpoint).
  - from_rollouts(): load already-saved rollout actions (offline; what compare_policies uses).

This is the LEARNED policy whose weights we improve via LoRA. It is compared against the fixed
policies, the paper-style plateau_then_w scheduler, and the oracle-sandwich rule (upper bound).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpt_oss"))
from lever_io import build_messages, parse_action  # noqa: E402


def from_rollouts(path):
    """Return {episode_id: action} from a rollout jsonl produced by rollout_base/rollout_adapter."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["episode_id"]] = r.get("action")
    return out


def live(trace_text, model=None, base_url=None):
    from client import chat
    raw = chat(build_messages(trace_text), model=model, base_url=base_url)
    action, reason, valid = parse_action(raw)
    return action, reason, valid, raw
