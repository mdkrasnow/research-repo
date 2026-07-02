"""Poll the Nebius Token Factory LoRA fine-tune job + print status, metrics, and the resulting
fine-tuned model id when done. Reads the job id from data/out/tf_finetune_job.json (or --job-id).

Env: NEBIUS_API_KEY (or GPT_OSS_API_KEY), GPT_OSS_BASE_URL.
Usage: python3 gpt_oss/poll_tf_finetune.py [--job-id ftjob-...] [--events]
"""

import argparse
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
JOB_FILE = os.path.join(HERE, "data", "out", "tf_finetune_job.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--events", action="store_true", help="also print recent training events")
    args = ap.parse_args()

    job_id = args.job_id
    if not job_id and os.path.exists(JOB_FILE):
        job_id = json.load(open(JOB_FILE)).get("job_id")
    if not job_id:
        sys.exit("no job id (pass --job-id or run launch_tf_finetune.py first)")

    key = os.getenv("GPT_OSS_API_KEY") or os.getenv("NEBIUS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base = os.getenv("GPT_OSS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/").rstrip("/")
    hdr = {"Authorization": f"Bearer {key}"}

    r = requests.get(f"{base}/fine_tuning/jobs/{job_id}", headers=hdr, timeout=60)
    r.raise_for_status()
    j = r.json()
    print(f"job        : {j.get('id')}")
    print(f"status     : {j.get('status')}")
    print(f"base model : {j.get('model')}")
    print(f"fine_tuned : {j.get('fine_tuned_model')}")
    print(f"trained_tok: {j.get('trained_tokens')}")
    if j.get("error"):
        print(f"error      : {j.get('error')}")

    if args.events:
        e = requests.get(f"{base}/fine_tuning/jobs/{job_id}/events", headers=hdr,
                         params={"limit": 10}, timeout=60)
        if e.ok:
            print("--- recent events ---")
            for ev in reversed(e.json().get("data", [])):
                print(f"  [{ev.get('level')}] {ev.get('message')}")


if __name__ == "__main__":
    main()
