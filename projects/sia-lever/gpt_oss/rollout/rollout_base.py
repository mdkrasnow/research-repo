"""Roll out the BASE gpt-oss-120b lever selector over eval traces.

For each eval episode: build messages, query the model, enforce strict JSON, retry ONCE with a
repair prompt if invalid. Save raw responses + parsed actions.

Output: results/gpt_oss/base_rollouts_<timestamp>.jsonl

Env: GPT_OSS_BASE_URL, GPT_OSS_API_KEY|NEBIUS_API_KEY|OPENAI_API_KEY, GPT_OSS_MODEL.
Use --limit to cap, --dry-run to skip API calls (emits null actions for plumbing tests).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lever_io import build_messages, parse_action, load_cache, PROJ  # noqa: E402

REPAIR = ("Your previous response was not valid JSON. Respond with ONLY this JSON object and "
          'nothing else: {"action": "<H|W|H_THEN_W|PROMOTE|KILL>", "reason": "<one sentence>"}')


def eval_episodes(cache, eval_seeds):
    seeds = sorted({r["seed"] for r in cache})
    keep = set(seeds[-eval_seeds:]) if eval_seeds else set(seeds)
    return [r for r in cache if r["seed"] in keep]


def rollout(model=None, base_url=None, limit=None, eval_seeds=3, dry_run=False, temperature=0.0,
            cache_path=None):
    cache = load_cache(cache_path)
    episodes = eval_episodes(cache, eval_seeds)
    if limit:
        episodes = episodes[:limit]

    from client import make_client, chat  # import ok without creds; calls need creds
    client = None
    if not dry_run:
        client = make_client(base_url=base_url)

    out = []
    for ep in episodes:
        msgs = build_messages(ep["trace_text"])
        raw, action, reason, valid = None, None, "", False
        if not dry_run:
            raw = chat(msgs, model=model, base_url=base_url, client=client, temperature=temperature)
            action, reason, valid = parse_action(raw)
            if action is None:  # one repair retry
                raw2 = chat(msgs + [{"role": "assistant", "content": raw or ""},
                                    {"role": "user", "content": REPAIR}],
                            model=model, base_url=base_url, client=client, temperature=temperature)
                action, reason, valid = parse_action(raw2)
                raw = (raw or "") + "\n---REPAIR---\n" + (raw2 or "")
        out.append({"episode_id": ep["episode_id"], "mode": ep["mode"], "seed": ep["seed"],
                    "raw_response": raw, "action": action, "reason": reason, "valid_json": valid})
        print(f"{ep['episode_id']}: action={action} valid_json={valid}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--cache", default=None, help="measured cache to roll out over (default: easy cache)")
    args = ap.parse_args()

    out = rollout(model=args.model, base_url=args.base_url, limit=args.limit,
                  eval_seeds=args.eval_seeds, dry_run=args.dry_run, cache_path=args.cache)
    d = os.path.join(PROJ, "results", "gpt_oss")
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(d, f"{args.tag}_rollouts_{stamp}.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nsaved {len(out)} rollouts -> {path}")


if __name__ == "__main__":
    main()
