"""Action parsing for lever-selector rollouts.

Thin re-export of the shared parser in gpt_oss/lever_io.py so the parser is identical across
data building, rollout, and eval. Importable both as a module and runnable for a quick check.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lever_io import parse_action, VALID_ACTIONS, action_json  # noqa: F401,E402

if __name__ == "__main__":
    tests = [
        '{"action":"H_THEN_W","reason":"x"}',
        '```json\n{"action": "W", "reason": "y"}\n```',
        'The answer is {"action": "H"} clearly',
        'I choose H_THEN_W here',
        'garbage no action',
    ]
    for t in tests:
        print(repr(t[:40]), "->", parse_action(t))
