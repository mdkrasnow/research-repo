"""Parallel lever-selector rollout — concurrent requests to an OpenAI-compatible endpoint (vLLM).

Same output schema as rollout_base.py (one jsonl row per episode: episode_id, mode, seed,
raw_response, action, reason, valid_json) but fires --concurrency requests at once. vLLM batches
them, so 24-48 reasoning-model calls drop from ~minutes (serial) to ~seconds.

Speedups baked in:
  - ThreadPoolExecutor over episodes (default 16 concurrent).
  - one repair retry per episode (same as base) but only on parse failure.
  - deterministic output order (sorted by episode_id) regardless of completion order.

Usage (through tunnel or on the VM):
  GPT_OSS_BASE_URL=http://localhost:8001/v1/ GPT_OSS_API_KEY=dummy \
  python gpt_oss/rollout/rollout_parallel.py --model lever_lora \
     --cache gpt_oss/data/out/hard_cache.jsonl --tag lora_fast --eval-seeds 1 --concurrency 16
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lever_io import build_messages, parse_action, load_cache, PROJ  # noqa: E402

REPAIR = ("Your previous response was not valid JSON. Respond with ONLY this JSON object and "
          'nothing else: {"action": "<H|W|H_THEN_W|PROMOTE|KILL>", "reason": "<one sentence>"}')


def eval_episodes(cache, eval_seeds):
    seeds = sorted({r["seed"] for r in cache})
    keep = set(seeds[-eval_seeds:]) if eval_seeds else set(seeds)
    return [r for r in cache if r["seed"] in keep]


def _one(ep, model, base_url, client, chat, temperature, max_tokens):
    msgs = build_messages(ep["trace_text"])
    raw = chat(msgs, model=model, base_url=base_url, client=client,
               temperature=temperature, max_tokens=max_tokens)
    action, reason, valid = parse_action(raw)
    if action is None:  # one repair retry
        raw2 = chat(msgs + [{"role": "assistant", "content": raw or ""},
                            {"role": "user", "content": REPAIR}],
                    model=model, base_url=base_url, client=client,
                    temperature=temperature, max_tokens=max_tokens)
        action, reason, valid = parse_action(raw2)
        raw = (raw or "") + "\n---REPAIR---\n" + (raw2 or "")
    return {"episode_id": ep["episode_id"], "mode": ep["mode"], "seed": ep["seed"],
            "raw_response": raw, "action": action, "reason": reason, "valid_json": valid}


def rollout(model=None, base_url=None, eval_seeds=1, temperature=0.0, max_tokens=1024,
            concurrency=16, cache_path=None, limit=None):
    cache = load_cache(cache_path)
    episodes = eval_episodes(cache, eval_seeds)
    if limit:
        episodes = episodes[:limit]
    from client import make_client, chat  # noqa: E402
    client = make_client(base_url=base_url)

    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_one, ep, model, base_url, client, chat, temperature, max_tokens)
                for ep in episodes]
        for fu in futs:
            r = fu.result()
            out.append(r)
            print(f"{r['episode_id']}: action={r['action']} valid={r['valid_json']}", flush=True)
    out.sort(key=lambda r: r["episode_id"])
    print(f"\n{len(out)} rollouts in {time.time()-t0:.1f}s (concurrency={concurrency})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--tag", default="fast")
    ap.add_argument("--eval-seeds", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = rollout(model=args.model, base_url=args.base_url, eval_seeds=args.eval_seeds,
                  max_tokens=args.max_tokens, concurrency=args.concurrency,
                  cache_path=args.cache, limit=args.limit)
    d = os.path.join(PROJ, "results", "gpt_oss")
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(d, f"{args.tag}_rollouts_{stamp}.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"saved {len(out)} -> {path}")


if __name__ == "__main__":
    main()
