"""Fail fast before a costly cluster run; does not inspect or select results."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

REQUIRED={"calibration_latents","calibration_labels","reference_images","endpoint_latents","endpoint_labels","endpoint_images","pairs"}
def main():
 p=argparse.ArgumentParser(); p.add_argument("bank",type=Path); args=p.parse_args()
 b=torch.load(args.bank,map_location="cpu",weights_only=True); missing=REQUIRED-set(b)
 if missing: raise SystemExit(f"missing bank tensors: {sorted(missing)}")
 pair=b["pairs"].long(); labels=b["endpoint_labels"]
 if pair.ndim!=2 or pair.shape[1]!=2 or len(pair)==0: raise SystemExit("pairs must be [N,2]")
 if len(torch.unique(pair)) != pair.numel(): raise SystemExit("endpoint reuse detected")
 if not torch.equal(labels[pair[:,0]],labels[pair[:,1]]): raise SystemExit("cross-class primary pairs forbidden")
 print(json.dumps({"status":"ready","pairs":len(pair),"reference":len(b["reference_images"]),"calibration":len(b["calibration_latents"]) }))
if __name__ == "__main__": main()
