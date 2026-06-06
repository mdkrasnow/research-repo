"""Shared lever-selector I/O: prompt loading, message construction, robust action parsing.

Used by data builders, rollout, and eval so the prompt and the parser are identical everywhere.
"""

import json
import os
import re

GPT_OSS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(GPT_OSS_DIR)

VALID_ACTIONS = ["H", "W", "H_THEN_W", "PROMOTE", "KILL"]
# Actions with a measured outcome in the cache (the rest map to documented outcomes in eval).
MEASURED_ACTIONS = ["H", "W", "H_THEN_W", "NOOP", "KILL"]

# The three real interventions that have measured-rerun outcomes.
ACTIVE_LEVERS = ["H", "W", "H_THEN_W"]
# Number of weight-retrains each lever costs (an unnecessary W is not free; see Phase 3).
W_RETRAINS = {"H": 0, "W": 1, "H_THEN_W": 1, "PROMOTE": 0, "KILL": 0, "NOOP": 0}
DEFAULT_W_COST = 0.05


def cost_adjusted_best(reward_by_action, w_cost=DEFAULT_W_COST, tol=1e-9):
    """Cost-adjusted argmax over the three active levers, tie-broken by preference order
    [H, W, H_THEN_W] (cheaper/simpler lever wins ties). Matches Phase 3 and the pre-registered
    correct lever. reward_by_action must contain H/W/H_THEN_W (real measured reruns)."""
    adj = {k: reward_by_action[k] - w_cost * W_RETRAINS[k] for k in ACTIVE_LEVERS}
    m = max(adj.values())
    for k in ACTIVE_LEVERS:
        if adj[k] >= m - tol:
            return k
    return ACTIVE_LEVERS[0]


def outcome_for_action(action, reward_by_action, w_cost=DEFAULT_W_COST):
    """Measured cost-adjusted outcome of a chosen action on a SIA-Lever episode.

    H/W/H_THEN_W -> their real measured reward minus W cost.
    PROMOTE/KILL  -> 0.0: every SIA-Lever episode is a genuine failure requiring intervention, so
                    declaring-solved or abandoning makes no real progress. Documented floor, not a
                    hand-authored per-mode delta."""
    if action in ACTIVE_LEVERS:
        return reward_by_action[action] - w_cost * W_RETRAINS[action]
    return 0.0


def regret_of_action(action, reward_by_action, w_cost=DEFAULT_W_COST):
    best = cost_adjusted_best(reward_by_action, w_cost)
    best_score = outcome_for_action(best, reward_by_action, w_cost)
    return best_score - outcome_for_action(action, reward_by_action, w_cost)


def load_prompts():
    with open(os.path.join(GPT_OSS_DIR, "prompts", "lever_selector_system.md")) as f:
        system = f.read().strip()
    with open(os.path.join(GPT_OSS_DIR, "prompts", "lever_selector_user_template.md")) as f:
        user_tmpl = f.read().strip()
    return system, user_tmpl


def build_messages(trace_text):
    system, user_tmpl = load_prompts()
    user = user_tmpl.replace("{trace_text}", trace_text)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_ACTION_RE = re.compile(r'"action"\s*:\s*"([A-Z_]+)"')


def parse_action(text):
    """Return (action, reason, valid_json). Robust to fences / extra prose.

    valid_json=False means we had to fall back to regex extraction (counts toward invalid_json_rate).
    """
    if text is None:
        return None, "", False
    s = text.strip()
    # strip ```json ... ``` fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    # try strict JSON (whole string, then first {...} block)
    for candidate in (s, _first_brace_block(s)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            act = str(obj.get("action", "")).upper().strip()
            if act in VALID_ACTIONS:
                return act, str(obj.get("reason", "")), True
        except (json.JSONDecodeError, AttributeError):
            pass
    # regex fallback
    m = _ACTION_RE.search(s)
    if m and m.group(1) in VALID_ACTIONS:
        return m.group(1), "", False
    # bare token fallback
    for a in sorted(VALID_ACTIONS, key=len, reverse=True):
        if re.search(rf"\b{a}\b", s):
            return a, "", False
    return None, "", False


def _first_brace_block(s):
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i:j + 1]
    return None


def action_json(action, reason=""):
    return json.dumps({"action": action, "reason": reason})


def load_cache(path=None):
    if path is None:
        path = os.path.join(PROJ, "gpt_oss", "data", "out", "action_outcome_cache.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
