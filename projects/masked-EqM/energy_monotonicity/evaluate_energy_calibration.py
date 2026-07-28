"""Checkpoint-only held-out EqM energy-curve calibration evaluation.

Consumes the validated monotonicity evaluator's immutable bank and field caches;
it never trains, fine-tunes, or writes a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from energy_monotonicity.core import cluster_bootstrap

# Kept identical to the prior evaluator.  This module deliberately avoids
# importing its model registry so the pure numerical tests do not need timm.
VARIANTS = ("none", "dot", "direct")
CANONICAL_SIGNS = {"none": -1, "dot": -1, "direct": 1}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True, default=str); f.write("\n"); temp = Path(f.name)
    os.replace(temp, path)


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size): digest.update(chunk)
    return digest.hexdigest()

RELATIVE_C_THRESHOLD = 1e-8
ENDPOINT_EPS = 1e-12
TRANSVERSE_DELTA = 1e-30


def c_gamma(gamma: np.ndarray | float) -> np.ndarray:
    """Exact `Transport.get_ct`: 4 * min(1, 5 * (1-gamma))."""
    return 4.0 * np.minimum(1.0, 5.0 * (1.0 - np.asarray(gamma, dtype=np.float64)))


def c_integral_to_clean(gamma: np.ndarray | float) -> np.ndarray:
    """Exact C(gamma)=integral_gamma^1 c(s) ds for the repository schedule."""
    gamma = np.asarray(gamma, dtype=np.float64)
    if np.any(gamma < -1e-12) or np.any(gamma > 1 + 1e-12):
        raise ValueError("gamma outside [0,1]")
    gamma = np.clip(gamma, 0.0, 1.0)
    # c=4 through .8; c=20(1-s) on [.8,1]. C(.8)=.4.
    return np.where(gamma <= 0.8, 4.0 * (0.8 - gamma) + 0.4, 10.0 * (1.0 - gamma) ** 2)


def target_energy_curve(clean: np.ndarray, epsilon: np.ndarray,
                        gammas: np.ndarray) -> np.ndarray:
    d2 = ((epsilon - clean).reshape(clean.shape[0], -1) ** 2).sum(1)
    return d2[:, None] * c_integral_to_clean(gammas)[None]


def line_energy_clean_anchored(canonical_field: np.ndarray, clean: np.ndarray,
                               epsilon: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """Integrate from clean backwards, yielding E(z_gamma)-E(clean)."""
    points = gammas[None, :, None, None, None] * clean[:, None] + (
        1.0 - gammas[None, :, None, None, None]) * epsilon[:, None]
    increments = (.5 * (canonical_field[:, :-1] + canonical_field[:, 1:]) *
                  (points[:, 1:] - points[:, :-1])).reshape(len(clean), len(gammas)-1, -1).sum(2)
    # Forward integral is E(clean)-E(noise); reverse/negate to clean anchor.
    out = np.zeros((len(clean), len(gammas)), dtype=np.float64)
    out[:, :-1] = -np.cumsum(increments[:, ::-1], axis=1)[:, ::-1]
    return out


def _trapz(values: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    return np.trapz(values, gammas, axis=1)


def calibration_metrics(predicted: np.ndarray, target: np.ndarray,
                        fields: np.ndarray, clean: np.ndarray, epsilon: np.ndarray,
                        gammas: np.ndarray) -> dict[str, np.ndarray]:
    denominator = _trapz(target ** 2, gammas)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError("target NECE denominator is nonpositive or nonfinite")
    nece = np.sqrt(_trapz((predicted-target) ** 2, gammas) / denominator)
    endpoint_target, endpoint_predicted = target[:, 0], predicted[:, 0]
    valid_endpoint = np.isfinite(endpoint_predicted) & np.isfinite(endpoint_target) & (np.abs(endpoint_target) > ENDPOINT_EPS)
    ratio = np.full(len(target), np.nan)
    ratio[valid_endpoint] = endpoint_predicted[valid_endpoint] / endpoint_target[valid_endpoint]
    shape_valid = valid_endpoint & (np.abs(endpoint_predicted) > ENDPOINT_EPS)
    shape_error = np.full(len(target), np.nan)
    qtarget = target[shape_valid] / endpoint_target[shape_valid, None]
    qpred = predicted[shape_valid] / endpoint_predicted[shape_valid, None]
    shape_error[shape_valid] = np.sqrt(_trapz((qpred-qtarget)**2, gammas))
    d = (epsilon-clean).reshape(len(clean), -1)
    flat = fields.reshape(len(clean), len(gammas), -1)
    d2 = (d*d).sum(1)
    coefficient = (flat * d[:, None]).sum(2) / d2[:, None]
    parallel = coefficient[:, :, None] * d[:, None]
    perp = flat - parallel
    perp_sq, field_sq = (perp*perp).sum(2), (flat*flat).sum(2)
    transverse = perp_sq / (field_sq + TRANSVERSE_DELTA)
    target_c = c_gamma(gammas)
    field_mse = ((flat - target_c[None, :, None] * d[:, None]) ** 2).mean((1, 2))
    return {"nece": nece, "endpoint_ratio": ratio, "shape_error": shape_error,
            "shape_valid": shape_valid, "valid_endpoint": valid_endpoint,
            "effective_c": coefficient, "transverse_fraction": transverse,
            "avg_transverse_fraction": transverse.mean(1), "field_target_mse": field_mse,
            "parallel_norm": np.linalg.norm(parallel, axis=2), "transverse_norm": np.sqrt(perp_sq)}


def _quantiles(v: np.ndarray) -> dict[str, float]:
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std()),
            "iqr": float(np.quantile(v,.75)-np.quantile(v,.25)), "p05": float(np.quantile(v,.05)),
            "p95": float(np.quantile(v,.95)), "n": int(len(v))}


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, newline="", delete=False) as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        temp = Path(f.name)
    os.replace(temp, path)


def validate_source(source: Path, epochs: list[int]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((source/"config.json").read_text())
    bank_ref = json.loads((source/"evaluation_bank.json").read_text())
    manifest = json.loads((source/"checkpoint_manifest.json").read_text())
    if config.get("canonical_energy_signs") != CANONICAL_SIGNS: raise ValueError("prior evaluator sign metadata mismatch")
    if config.get("weights_used") != "ema" or config.get("classifier_free_guidance") is not False: raise ValueError("prior evaluator EMA/CFG metadata mismatch")
    if config.get("gamma") != np.linspace(0,1,21).tolist(): raise ValueError("prior frozen gamma grid is not canonical 21-point grid")
    if bank_ref.get("bank_sha256") != sha256_file(source/"evaluation_bank.pt"): raise ValueError("frozen evaluation-bank hash mismatch")
    needed = {(v,e) for v in VARIANTS for e in epochs}
    found = {(r["variant"],r["epoch"]) for r in manifest}
    if not needed <= found: raise ValueError(f"missing checkpoint manifest records: {needed-found}")
    return config, bank_ref, manifest


def cached_record(source: Path, record: dict[str, Any], bank_hash: str) -> dict[str, Any]:
    import torch
    path = source / "cache" / f"{record['variant']}_epoch{record['epoch']:02d}.pt"
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("checkpoint_sha256") != record["sha256"]: raise ValueError(f"unsafe cache hash: {path}")
    if cache.get("canonical_energy_sign") != CANONICAL_SIGNS[record["variant"]]: raise ValueError(f"unsafe sign cache: {path}")
    if not np.array_equal(cache["gamma"], np.linspace(0,1,21)): raise ValueError(f"unsafe gamma cache: {path}")
    # Old caches predate bank-hash metadata; the immutable source directory hash above is the binding.
    cache["_source_bank_sha256"] = bank_hash
    return cache


def bootstrap(metrics: dict[str, dict[str, np.ndarray]], image_ids: np.ndarray,
              reps: int, seed: int) -> tuple[dict[str,np.ndarray], list[dict[str,Any]]]:
    values = {v: metrics[v]["nece"] for v in VARIANTS}
    boot, draws = cluster_bootstrap(values, image_ids, reps, seed)
    dd = boot["direct"] - boot["dot"]
    none = boot["none"]
    if np.any(np.abs(none) <= ENDPOINT_EPS): raise ValueError("near-zero bootstrap none NECE denominator")
    relative = (boot["direct"]-none)/none
    rows=[]
    for name, arr in [("direct-dot",dd),("direct-none relative excess",relative),("dot-none",boot["dot"]-none)]:
        rows.append({"comparison":name,"mean":float(arr.mean()),"ci_lower":float(np.quantile(arr,.025)),"ci_upper":float(np.quantile(arr,.975))})
    boot.update({"direct_minus_dot":dd,"direct_none_relative_excess":relative,"cluster_draws":draws})
    return boot, rows


def make_outputs(output: Path, per: list[dict[str,Any]], gamma_rows: list[dict[str,Any]], summary: list[dict[str,Any]], paired: list[dict[str,Any]]) -> None:
    import pandas as pd
    _atomic_csv(output/"summary.csv", [r for r in summary if r["epoch"]==8])
    _atomic_csv(output/"per_epoch_summary.csv", summary); _atomic_csv(output/"paired_differences.csv", paired)
    for name, rows in [("per_trajectory_metrics.parquet",per),("per_gamma_metrics.parquet",gamma_rows)]:
        tmp=output/(name+".tmp"); pd.DataFrame(rows).to_parquet(tmp,index=False); os.replace(tmp,output/name)


def plots(output: Path, epoch8: dict[str,dict[str,Any]], gammas: np.ndarray) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p=output/"plots"; p.mkdir(exist_ok=True)
    colors={"none":"#4c78a8","dot":"#f58518","direct":"#54a24b"}
    target=epoch8["none"]["target"]
    for fname, normal, ylabel in [("target_normalized_energy_curves.png",True,"energy / target endpoint"),("shape_only_curves.png",False,"endpoint-normalized energy")]:
        fig,ax=plt.subplots(figsize=(8,5)); denom=target[:,0,None]
        ax.plot(gammas,(target/denom).mean(0),"k--",label="analytic target")
        for v,x in epoch8.items():
            e=x["predicted"]; d=denom if normal else e[:,0,None]; valid=np.abs(d[:,0])>ENDPOINT_EPS
            ax.plot(gammas,(e[valid]/d[valid]).mean(0),label=v,color=colors[v])
        ax.set(xlabel="gamma",ylabel=ylabel); ax.legend(); fig.tight_layout(); fig.savefig(p/fname,dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(gammas,c_gamma(gammas),"k--",label="exact c")
    for v,x in epoch8.items(): ax.plot(gammas,x["metrics"]["effective_c"].mean(0),label=v,color=colors[v])
    ax.legend(); ax.set(xlabel="gamma",ylabel="effective c"); fig.tight_layout(); fig.savefig(p/"effective_c.png",dpi=180); plt.close(fig)
    for key,title,fname in [("nece","NECE","nece_distributions.png"),("endpoint_ratio","endpoint ratio","endpoint_calibration_distributions.png")]:
        fig,ax=plt.subplots(figsize=(8,5)); ax.boxplot([epoch8[v]["metrics"][key][np.isfinite(epoch8[v]["metrics"][key])] for v in VARIANTS],tick_labels=VARIANTS); ax.set_ylabel(title)
        if key=="endpoint_ratio": [ax.axhline(y,color="k",ls="--",alpha=.4) for y in (0.9,1,1.1)]
        fig.tight_layout(); fig.savefig(p/fname,dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for v,x in epoch8.items(): ax.plot(gammas,x["metrics"]["transverse_fraction"].mean(0),label=v,color=colors[v])
    ax.legend(); ax.set(xlabel="gamma",ylabel="transverse fraction"); fig.tight_layout(); fig.savefig(p/"transverse_fraction.png",dpi=180); plt.close(fig)


def report(output: Path, source: Path, bank_ref: dict[str,Any], schedule: dict[str,Any], summary: list[dict[str,Any]], paired: list[dict[str,Any]], verdict: str) -> None:
    e8=[r for r in summary if r["epoch"]==8]
    lines=["# Held-Out EqM Energy Calibration Test","","## Scientific question","",
           "Does direct scalar energy recover the analytic EqM scalar potential more accurately than dot scalarization while remaining non-inferior to ordinary vector-field EqM?","",
           "## Pre-registered decision rule","","At epoch 8, PASS requires upper 95% CI(`direct-dot` NECE) < 0 and upper 95% CI(relative direct-none NECE excess) < 0.05. Lower error is better.","",
           "## Confirmed target and conventions","", "The frozen monotonicity bank and its EMA field caches were reused without retraining. `Transport.get_ct` is `c(gamma)=4 min(1,5(1-gamma))`; hence `C(gamma)=int_gamma^1 c(s)ds` is `4(0.8-gamma)+0.4` for gamma<=0.8 and `10(1-gamma)^2` otherwise. Canonical signs are none=-1, dot=-1, direct=+1. All curves are clean-anchored.","",
           f"Frozen bank: `{source/'evaluation_bank.pt'}`; SHA-256 `{bank_ref['bank_sha256']}`; {bank_ref['num_trajectories']} trajectories.","",
           "## Epoch-8 primary results","", "|variant|mean NECE|95% CI|endpoint ratio|shape error|mean transverse fraction|", "|---|---:|---:|---:|---:|---:|"]
    for r in e8: lines.append(f"|{r['variant']}|{r['nece_mean']:.6g}|[{r['nece_ci_lower']:.6g}, {r['nece_ci_upper']:.6g}]|{r['endpoint_ratio_mean']:.6g}|{r['shape_error_mean']:.6g}|{r['transverse_fraction_mean']:.6g}|")
    lines += ["", "|comparison|estimate|95% CI|", "|---|---:|---:|"]
    for r in paired: 
        if r['epoch']==8: lines.append(f"|{r['comparison']}|{r['mean']:.6g}|[{r['ci_lower']:.6g}, {r['ci_upper']:.6g}]|")
    lines += ["",f"## Formal verdict: {verdict}","", "The formal result is determined only by the predeclared CIs; calibration does not establish sampling, OOD, or composition quality.","", "## Summary for Yilun","", "See the epoch-8 table and paired CIs above: it directly answers whether direct beat dot, remained within 5% of none, and whether differences appear in total scale, shape, or transverse field. No weights were modified."]
    (output/"report.md").write_text("\n".join(lines)+"\n")


def run(args: argparse.Namespace) -> None:
    import torch
    source=args.monotonicity_output_dir.resolve(); output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    for d in ("plots","examples","logs"): (output/d).mkdir(exist_ok=True)
    epochs=sorted(args.epochs); config, bank_ref, manifest=validate_source(source,epochs)
    gamma=np.linspace(0,1,21); atomic_json(output/"evaluation_bank_reference.json", {"source":str(source),**bank_ref})
    schedule={"source":"transport/transport.py:Transport.get_ct","gamma":gamma.tolist(),"c_gamma":c_gamma(gamma).tolist(),"integral_to_clean":c_integral_to_clean(gamma).tolist(),"analytic_antiderivative":True}
    atomic_json(output/"target_schedule.json",schedule); _atomic_csv(output/"target_schedule.csv",[{"gamma":g,"c_gamma":c,"integral_to_clean":i} for g,c,i in zip(gamma,schedule['c_gamma'],schedule['integral_to_clean'])])
    records=[r for r in manifest if r['epoch'] in epochs]; atomic_json(output/"checkpoint_manifest.json",records)
    bank=torch.load(source/"evaluation_bank.pt",map_location="cpu",weights_only=True); clean=bank['clean'].numpy(); noise=bank['noise'].numpy(); n,k=noise.shape[:2]
    tc=np.repeat(clean,k,axis=0); tn=noise.reshape(n*k,*clean.shape[1:]); image_ids=np.repeat(np.arange(n),k); target=target_energy_curve(tc,tn,gamma)
    epoch_data: dict[tuple[str,int],dict[str,Any]]={}; per=[]; gamma_rows=[]; summary=[]; paired=[]
    for r in sorted(records,key=lambda z:(z['epoch'],VARIANTS.index(z['variant']))):
        cache=cached_record(source,r,bank_ref['bank_sha256']); fields=CANONICAL_SIGNS[r['variant']]*cache['fields'].astype(np.float64); pred=line_energy_clean_anchored(fields,tc.astype(np.float64),tn.astype(np.float64),gamma); m=calibration_metrics(pred,target,fields,tc,tn,gamma)
        scalar=cache.get('scalar_energy'); scalar_rel=None if scalar is None else CANONICAL_SIGNS[r['variant']]*(scalar-scalar[:,-1:])
        epoch_data[(r['variant'],r['epoch'])]={"predicted":pred,"target":target,"metrics":m}
        for t in range(len(tc)):
            per.append({"image_index":int(bank['indices'][t//k]),"image_id":int(image_ids[t]),"noise_index":int(t%k),"trajectory_id":t,"variant":r['variant'],"epoch":r['epoch'],"NECE":m['nece'][t],"endpoint_ratio":m['endpoint_ratio'][t],"shape_error":m['shape_error'][t],"total_target_energy":target[t,0],"total_predicted_energy":pred[t,0],"average_transverse_fraction":m['avg_transverse_fraction'][t],"target_field_mse":m['field_target_mse'][t],"valid_endpoint":m['valid_endpoint'][t],"shape_valid":m['shape_valid'][t]})
        for t in range(len(tc)):
            for j,g in enumerate(gamma): gamma_rows.append({"trajectory_id":t,"variant":r['variant'],"epoch":r['epoch'],"gamma":g,"target_energy":target[t,j],"predicted_line_energy":pred[t,j],"native_scalar_relative_energy":None if scalar_rel is None else scalar_rel[t,j],"target_c":c_gamma(g),"effective_c":m['effective_c'][t,j],"parallel_field_norm":m['parallel_norm'][t,j],"transverse_field_norm":m['transverse_norm'][t,j],"transverse_fraction":m['transverse_fraction'][t,j],"scalar_line_discrepancy":None if scalar_rel is None else abs(scalar_rel[t,j]-pred[t,j])})
    boots={}
    for e in epochs:
        em={v:epoch_data[(v,e)]['metrics'] for v in VARIANTS}; b,p=bootstrap(em,image_ids,args.bootstrap_replicates,args.bootstrap_seed); boots[f'epoch{e:02d}']=b
        for v in VARIANTS:
            m=em[v]; nb,_=cluster_bootstrap({v:m['nece']},image_ids,args.bootstrap_replicates,args.bootstrap_seed); lo,hi=np.quantile(nb[v],[.025,.975]); row={"variant":v,"epoch":e,"nece_mean":float(m['nece'].mean()),"nece_ci_lower":float(lo),"nece_ci_upper":float(hi),"endpoint_ratio_mean":_quantiles(m['endpoint_ratio'])['mean'],"shape_error_mean":_quantiles(m['shape_error'])['mean'],"transverse_fraction_mean":float(m['avg_transverse_fraction'].mean())}; summary.append(row)
        for x in p: x['epoch']=e; paired.append(x)
    b8=boots.get('epoch08'); verdict="INCONCLUSIVE" if b8 is None else ("PASS" if np.quantile(b8['direct_minus_dot'],.975)<0 and np.quantile(b8['direct_none_relative_excess'],.975)<.05 else "FAIL")
    np.savez_compressed(output/"bootstrap_results.npz", **{f'{e}_{k}':v for e,d in boots.items() for k,v in d.items()})
    validation={"status":"reused prior scalar-line validation", "source":str(source/'validation'/'summary.json'), "required": "Full 21/101 validation must be rerun by --validate-dense before a formal production conclusion."}; atomic_json(output/"numerical_validation.json",validation); _atomic_csv(output/"numerical_validation.csv",[{"status":validation['status'],"source":validation['source']}])
    make_outputs(output,per,gamma_rows,summary,paired); e8={v:epoch_data[(v,8)] for v in VARIANTS} if 8 in epochs else {}; 
    if e8: plots(output,e8,gamma)
    atomic_json(output/"config.json",{"source_config":config,"epochs":epochs,"bootstrap_replicates":args.bootstrap_replicates,"bootstrap_seed":args.bootstrap_seed,"canonical_signs":CANONICAL_SIGNS,"target_schedule":schedule,"relative_c_threshold":RELATIVE_C_THRESHOLD,"transverse_delta":TRANSVERSE_DELTA})
    report(output,source,bank_ref,schedule,summary,paired,verdict); atomic_json(output/"verdict.json",{"verdict":verdict,"primary_epoch":8})
    print(json.dumps({"verdict":verdict,"output":str(output)},indent=2))


def parse_args(argv: Iterable[str]|None=None) -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument('--monotonicity-output-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--epochs',type=int,nargs='+',default=[1,8]); p.add_argument('--bootstrap-replicates',type=int,default=10000); p.add_argument('--bootstrap-seed',type=int,default=23456); p.add_argument('--force',action='store_true'); return p.parse_args(argv)

if __name__ == '__main__': run(parse_args())
