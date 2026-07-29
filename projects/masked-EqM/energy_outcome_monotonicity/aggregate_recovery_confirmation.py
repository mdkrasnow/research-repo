"""Fail-closed aggregation for the preregistered three-seed recovery test."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def seed_effect(npz: Path, *, bootstrap: int, seed: int) -> tuple[float, list[float]]:
    z=np.load(npz); required={"dot_error","direct_error"}
    if required-set(z.files): raise ValueError(f"{npz} missing {required-set(z.files)}")
    dot,direct=z["dot_error"],z["direct_error"]
    if dot.shape != direct.shape or dot.ndim != 2: raise ValueError("expected paired [image,candidate] arrays")
    effect=dot.mean(1)-direct.mean(1); rng=np.random.default_rng(seed)
    boot=effect[rng.integers(0,len(effect),(bootstrap,len(effect)))].mean(1)
    return float(effect.mean()),np.quantile(boot,[.025,.975]).tolist()

def main(a):
    if len(a.reports)!=3: raise ValueError("confirmation requires exactly three predeclared seed reports")
    effects=[]; rows=[]
    for i,p in enumerate(a.reports):
        mean,ci=seed_effect(p,bootstrap=a.bootstrap,seed=a.seed+i); effects.append(mean); rows.append({"report":str(p),"dot_minus_direct":mean,"ci95":ci})
    x=np.asarray(effects); se=x.std(ddof=1)/np.sqrt(3); t=x.mean()/se if se else float("inf")
    # Exact two-sided t(2) p-value is 1 - |t|/sqrt(t^2+2).
    p=1-abs(t)/np.sqrt(t*t+2) if np.isfinite(t) else 0.0
    out={"seed_rows":rows,"mean_effect":float(x.mean()),"t_stat":float(t),"two_sided_p":float(p),"verdict":"PASS" if x.mean()>0 and p<.05 and bool((x>0).all()) else "FAIL"}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+"\n")
if __name__ == "__main__":
 p=argparse.ArgumentParser(); p.add_argument("--reports",type=Path,nargs="+",required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--bootstrap",type=int,default=10000); p.add_argument("--seed",type=int,default=20260801); main(p.parse_args())
