"""Download a Token Factory fine-tuning job's result_files (LoRA adapter shards) to disk.

Token Factory does NOT auto-serve fine-tuned gpt-oss on the shared endpoint (custom-weights
hosting is beta / on-request as of 2026-06). So we preserve the adapter artifacts here; they can
later be served on a GPU VM (vLLM + base) or via beta custom-weights once access lands.

Usage: python3 gpt_oss/download_ft_result.py --job-id ftjob-... --tag <tag>
       (or --job-file gpt_oss/data/out/tf_finetune_job_<tag>.json)
Env: NEBIUS_API_KEY (or GPT_OSS_API_KEY), GPT_OSS_BASE_URL.
"""

import argparse
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--job-file", default=None)
    ap.add_argument("--tag", default="ft")
    args = ap.parse_args()

    job_id = args.job_id
    if not job_id and args.job_file:
        job_id = json.load(open(args.job_file))["job_id"]
    if not job_id:
        sys.exit("need --job-id or --job-file")

    key = os.getenv("GPT_OSS_API_KEY") or os.getenv("NEBIUS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base = os.getenv("GPT_OSS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/").rstrip("/")
    hdr = {"Authorization": f"Bearer {key}"}

    j = requests.get(f"{base}/fine_tuning/jobs/{job_id}", headers=hdr, timeout=60).json()
    files = j.get("result_files", [])
    if not files:
        sys.exit(f"no result_files (status={j.get('status')})")

    out_dir = os.path.join(HERE, "..", "adapters", "gpt_oss_120b", f"{args.tag}_{job_id}")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"job {job_id} status={j.get('status')} -> {len(files)} files -> {out_dir}")

    manifest = {"job_id": job_id, "tag": args.tag, "model": j.get("model"),
                "trained_tokens": j.get("trained_tokens"), "files": []}
    for fid in files:
        meta = requests.get(f"{base}/files/{fid}", headers=hdr, timeout=60).json()
        name = meta.get("filename", fid)
        r = requests.get(f"{base}/files/{fid}/content", headers=hdr, timeout=300)
        r.raise_for_status()
        dst = os.path.join(out_dir, os.path.basename(name))
        with open(dst, "wb") as f:
            f.write(r.content)
        manifest["files"].append({"id": fid, "name": name, "bytes": len(r.content)})
        print(f"  {name}  ({len(r.content)} B)")

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved manifest -> {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
