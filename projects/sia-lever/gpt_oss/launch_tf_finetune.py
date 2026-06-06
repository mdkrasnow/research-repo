"""Launch a REAL managed LoRA fine-tune of gpt-oss-120b on Nebius Token Factory.

Token Factory exposes an OpenAI-compatible fine-tuning API (verified live 2026-06-06):
  POST /v1/files            (purpose="fine-tune")  -> file id
  POST /v1/fine_tuning/jobs (model + training_file + hyperparameters{lora:true,...})

This runs the lever-selector SFT on Nebius H200s — the real GPU experiment. The local SFT
datasets carry extra bookkeeping keys (episode_id, reward_by_action, ...); OpenAI-format upload
wants ONLY {"messages": [...]} per line, so we strip first.

Env: NEBIUS_API_KEY (or GPT_OSS_API_KEY), GPT_OSS_BASE_URL, GPT_OSS_MODEL.
Usage: python3 gpt_oss/launch_tf_finetune.py [--dry-run]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "out")
TMP = os.path.join(HERE, "data", "out", "_tf_upload")


def _clean(src, dst):
    """Strip every line to just {"messages": [...]} for OpenAI-format upload."""
    n = 0
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fo.write(json.dumps({"messages": obj["messages"]}) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="prep files, skip upload + job create")
    ap.add_argument("--model", default=os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-120b"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=None, help="override TF default (8); smaller = more steps")
    ap.add_argument("--packing", default=None, choices=["true", "false"],
                    help="false => one example per step (epochs actually count on tiny data)")
    ap.add_argument("--learning-rate", type=float, default=None, help="override TF default (1e-5)")
    ap.add_argument("--tag", default="sft", help="rung tag: names upload files, job-suffix, job-out json")
    ap.add_argument("--src-train", default=os.path.join(DATA, "trace_action_train.jsonl"))
    ap.add_argument("--src-eval", default=os.path.join(DATA, "trace_action_eval.jsonl"))
    ap.add_argument("--job-out", default=None, help="path to write job json (default: data/out/tf_finetune_job_<tag>.json)")
    args = ap.parse_args()

    os.makedirs(TMP, exist_ok=True)
    train_clean = os.path.join(TMP, f"sft_train_{args.tag}.jsonl")
    eval_clean = os.path.join(TMP, f"sft_eval_{args.tag}.jsonl")
    n_tr = _clean(args.src_train, train_clean)
    n_ev = _clean(args.src_eval, eval_clean)
    print(f"prepared train={n_tr} eval={n_ev} (model={args.model}, lora_r={args.lora_r}, epochs={args.epochs})")

    if args.dry_run:
        print("dry-run: stopping before upload/job-create")
        return

    import requests
    key = os.getenv("GPT_OSS_API_KEY") or os.getenv("NEBIUS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base = os.getenv("GPT_OSS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/").rstrip("/")
    if not key:
        sys.exit("No API key. Set NEBIUS_API_KEY.")
    hdr = {"Authorization": f"Bearer {key}"}

    def _upload(path):
        with open(path, "rb") as fh:
            r = requests.post(f"{base}/files", headers=hdr,
                              files={"file": (os.path.basename(path), fh, "application/jsonl")},
                              data={"purpose": "fine-tune"}, timeout=120)
        r.raise_for_status()
        return r.json()["id"]

    print("uploading training file...")
    tr_id = _upload(train_clean)
    print(f"  train file id = {tr_id}")
    print("uploading validation file...")
    ev_id = _upload(eval_clean)
    print(f"  eval  file id = {ev_id}")

    print("creating fine-tuning job (LoRA)...")
    # raw POST so Nebius-specific lora_* hyperparameters transmit verbatim (not in OpenAI schema)
    payload = {
        "model": args.model,
        "training_file": tr_id,
        "validation_file": ev_id,
        "suffix": f"sia-lever-{args.tag}",
        "hyperparameters": {
            "n_epochs": args.epochs,
            "lora": True,
            "lora_r": args.lora_r,
            "lora_alpha": 2 * args.lora_r,
        },
    }
    if args.batch_size is not None:
        payload["hyperparameters"]["batch_size"] = args.batch_size
    if args.packing is not None:
        payload["hyperparameters"]["packing"] = (args.packing == "true")
    if args.learning_rate is not None:
        payload["hyperparameters"]["learning_rate"] = args.learning_rate
    r = requests.post(f"{base}/fine_tuning/jobs", headers={**hdr, "Content-Type": "application/json"},
                      json=payload, timeout=120)
    if r.status_code >= 300:
        sys.exit(f"job create failed HTTP {r.status_code}: {r.text}")
    job = r.json()
    print(f"\nJOB CREATED: id={job.get('id')} status={job.get('status')} model={job.get('model')}")
    print("poll: python3 gpt_oss/poll_tf_finetune.py")
    # persist job id for tracking
    rec = {"job_id": job.get("id"), "status": job.get("status"), "model": job.get("model"),
           "tag": args.tag, "train_file": tr_id, "eval_file": ev_id, "n_train": n_tr, "n_eval": n_ev}
    out = args.job_out or os.path.join(HERE, "data", "out", f"tf_finetune_job_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
