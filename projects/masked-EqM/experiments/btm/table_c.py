import json,glob,statistics as st
ARM={"btm_IIB_V_s0":"V  vector BTM","btm_IIB_G_s0":"G  scalar exact",
     "btm_IIB_D1_s0":"D1 FD directional K=1","btm_IIB_D4_s0":"D4 FD directional K=4",
     "btm_IIB_LEGACYvec_s0":"LEGACY vector (neg ctrl)","btm_IIB_LEGACYscalar_s0":"LEGACY direct scalar (neg ctrl)"}
order=list(ARM)
def cv(x):
    m=st.mean(x); return st.pstdev(x)/m if m else float("nan")
data={}
for d in sorted(glob.glob("*/*/gradient_metrics.jsonl")):
    tag=d.split("/")[0].rsplit("_job",1)[0]
    G=[];P=[];E=[]
    for l in open(d):
        r=json.loads(l)
        (G if "grad_norm" in r else P if "delta_theta_norm" in r else E if "E_mean" in r else []).append(r)
    data[tag]=(G,P,E)

print("TABLE C -- Phase II-B optimizer behaviour (ImageNet B/2, global batch 256, tc=0.9, 4xA100 DDP)")
print("late = final 20% of each run's logged steps.  clip% over all gradient records.\n")
h=(f"{'arm':32s} {'grec':>5s} {'maxstep':>8s} {'g med':>8s} {'g CV':>7s} {'g p95':>8s} "
   f"{'g max':>9s} {'late med':>9s} {'late CV':>8s} {'clip%':>6s}")
print(h); print("-"*len(h))
for tag in order:
    if tag not in data: continue
    G,P,E=data[tag]
    if not G: continue
    g=[r["grad_norm"] for r in G]; s=sorted(g)
    cut=int(len(g)*0.8); late=g[cut:]
    print(f"{ARM[tag]:32s} {len(G):5d} {max(r['step'] for r in G):8d} {st.median(g):8.4f} {cv(g):7.3f} "
          f"{s[int(len(s)*.95)]:8.3f} {max(g):9.3f} {st.median(late):9.4f} {cv(late):8.3f} "
          f"{100*sum(bool(r.get('clipped')) for r in G)/len(G):6.2f}")

print("\nFD estimator health:")
for tag in order:
    if tag not in data: continue
    G=data[tag][0]
    fg=[r["fd_gap_abs"] for r in G if r.get("fd_gap_abs") is not None]
    if not fg: continue
    fh=[r["fd_h_mean"] for r in G if r.get("fd_h_mean") is not None]
    fr=[r["fd_target_rms"] for r in G if r.get("fd_target_rms") is not None]
    m=lambda x: f"{st.median(x):.4g}" if x else "n/a"
    print(f"  {ARM[tag]:32s} fd_gap med {st.median(fg):.3e}   h med {m(fh)}   target_rms med {m(fr)}")

print("\nWeak-conservation / step probes (P_t, delta_theta_norm, eta_func):")
for tag in order:
    if tag not in data: continue
    P=data[tag][1]
    if not P: continue
    ks=[k for k in ("P_t","delta_theta_norm","eta_func") if k in P[0]]
    out=" ".join(f"{k} med {st.median([r[k] for r in P if r.get(k) is not None]):.4g}" for k in ks)
    print(f"  {ARM[tag]:32s} n={len(P):4d}  {out}")

print("\nEnergy / potential diagnostics (scalar arms only):")
for tag in order:
    if tag not in data: continue
    E=data[tag][2]
    if not E: continue
    print(f"  {ARM[tag]:32s} n={len(E):4d}  E_mean med {st.median([r['E_mean'] for r in E]):+.4g}  "
          f"E_std med {st.median([r['E_std'] for r in E]):.4g}")

print("\n" + "="*110)
print("STEP-MATCHED PANEL -- all arms restricted to the SAME absolute window.")
print("The panel above compares different horizons (legacy arms were killed early by EDQUOT,")
print("D1 died at 66.9k, D4 was cancelled at 11.7k), so its 'late' columns are NOT comparable.")
print("Absolute windows make them comparable at the cost of horizon.\n")
for lo,hi in ((0,11700),(0,22800),(0,54300),(0,66900),(0,75000)):
    print(f"--- window [{lo}, {hi}] steps ---")
    hh=f"{'arm':32s} {'n':>5s} {'g med':>8s} {'g CV':>7s} {'g p95':>8s} {'g max':>9s} {'P_t med':>10s}"
    print(hh)
    for tag in order:
        if tag not in data: continue
        G,P,E=data[tag]
        g=[r["grad_norm"] for r in G if lo<=r["step"]<=hi]
        if len(g)<20: continue
        s=sorted(g)
        pt=[r["P_t"] for r in P if lo<=r.get("step",-1)<=hi and r.get("P_t") is not None]
        ptm=f"{st.median(pt):10.4g}" if pt else f"{'n/a':>10s}"
        print(f"{ARM[tag]:32s} {len(g):5d} {st.median(g):8.4f} {cv(g):7.3f} {s[int(len(s)*.95)]:8.3f} {max(g):9.3f} {ptm}")
    print()
