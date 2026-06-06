"""Roll out a gpt-oss-120b + LoRA adapter lever selector over eval traces.

Two backends:
  --endpoint  : query an OpenAI-compatible server that already serves base+adapter (recommended;
                serve with gpt_oss/serve/serve_vllm_adapter.sh, then this is identical to
                rollout_base but tagged for the adapter). Uses GPT_OSS_BASE_URL/model.
  --local     : load base model + PEFT adapter with transformers and generate locally (heavy for
                120B; use for small models or when no server is available).

Output: results/gpt_oss/<tag>_rollouts_<timestamp>.jsonl
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJ, "gpt_oss"))
from lever_io import build_messages, parse_action, load_cache  # noqa: E402


def eval_episodes(cache, eval_seeds):
    seeds = sorted({r["seed"] for r in cache})
    keep = set(seeds[-eval_seeds:]) if eval_seeds else set(seeds)
    return [r for r in cache if r["seed"] in keep]


def local_generator(base_model, adapter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def gen(messages):
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=128, do_sample=False)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="adapter dir (for --local) or label (for --endpoint)")
    ap.add_argument("--base-model", default=os.getenv("GPT_OSS_MODEL_PATH", "openai/gpt-oss-120b"))
    ap.add_argument("--endpoint", action="store_true", help="query a served base+adapter endpoint")
    ap.add_argument("--local", action="store_true", help="load base+adapter locally via PEFT")
    ap.add_argument("--model", default=None, help="served model id for --endpoint")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="sft")
    args = ap.parse_args()

    cache = load_cache()
    episodes = eval_episodes(cache, args.eval_seeds)
    if args.limit:
        episodes = episodes[:args.limit]

    if args.local:
        gen = local_generator(args.base_model, args.adapter)
        def ask(msgs):
            return gen(msgs)
    else:  # endpoint
        from client import make_client, chat
        client = make_client(base_url=args.base_url)
        def ask(msgs):
            return chat(msgs, model=args.model, base_url=args.base_url, client=client)

    out = []
    for ep in episodes:
        raw = ask(build_messages(ep["trace_text"]))
        action, reason, valid = parse_action(raw)
        out.append({"episode_id": ep["episode_id"], "mode": ep["mode"], "seed": ep["seed"],
                    "raw_response": raw, "action": action, "reason": reason, "valid_json": valid,
                    "adapter": args.adapter})
        print(f"{ep['episode_id']}: action={action} valid_json={valid}", flush=True)

    d = os.path.join(PROJ, "results", "gpt_oss")
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(d, f"{args.tag}_rollouts_{stamp}.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nsaved {len(out)} adapter rollouts -> {path}")


if __name__ == "__main__":
    main()
