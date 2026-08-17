import json,statistics as st
d=json.load(open("results/btm/toy_summary.json"))["summary"]
NAME={"btm_vector":"V   vector BTM","btm_scalar_exact":"G   scalar exact grad-match",
 "btm_scalar_action_exact":"A   scalar Action/Ritz exact","btm_scalar_fd_directional":"D   scalar FD directional",
 "btm_scalar_fd_action":"F   scalar FD Action/Ritz",
 "legacy_vector":"LEGACY vector (neg ctrl)","legacy_scalar":"LEGACY direct scalar (neg ctrl)",
 "legacy_direct_vector":"LEGACY vector (neg ctrl)","legacy_direct_scalar":"LEGACY direct scalar (neg ctrl)",
 "eqm_legacy_vector":"LEGACY vector  (neg ctrl)","eqm_legacy_scalar":"LEGACY direct scalar (neg ctrl)"}
rows={}
for k,v in d.items():
    arm,K,eps,geo=k.split("|")
    rows.setdefault((arm,K,eps),{})[geo]=v
order=["btm_vector","btm_scalar_exact","btm_scalar_action_exact",
       "btm_scalar_fd_directional","btm_scalar_fd_action"]
def rank(t):
    a=t[0]
    return (order.index(a) if a in order else 99, t[1], t[2])

print("TABLE A -- toy 5-atom transport (10 seeds/arm, >=100k fresh x0, tc=0.9)")
print("primary metric = basin-mass MAE (lower better); R_psi = weak conservation residual\n")
h=(f"{'arm':34s} {'K':>3s} {'eps':>7s} | {'ring MAE':>9s} {'+/-':>7s} {'R_psi':>7s} "
   f"| {'asym MAE':>9s} {'+/-':>7s} {'R_psi':>7s} | {'stab':>5s}")
print(h); print("-"*len(h))
V={}
for key in sorted(rows,key=rank):
    arm,K,eps=key; g=rows[key]
    r=g.get("ring"); a=g.get("asym")
    if not r or not a: continue
    if arm=="btm_vector" and K=="K1": V={"ring":r["median"],"asym":a["median"]}
    nm=NAME.get(arm,arm)
    print(f"{nm:34s} {K[1:]:>3s} {eps[3:]:>7s} | {r['median']:9.4f} {r['std']:7.4f} {r['weak_residual']:7.4f} "
          f"| {a['median']:9.4f} {a['std']:7.4f} {a['weak_residual']:7.4f} | "
          f"{min(r['n_stable'],a['n_stable']):2d}/{r['n']:2d}")

print("\nPRE-REGISTERED GATE: MAE_D <= max(0.015, 2 x MAE_V), both geometries.")
for key in sorted(rows,key=rank):
    arm,K,eps=key
    if not arm.startswith("btm_scalar"): continue
    g=rows[key]
    if "ring" not in g or "asym" not in g: continue
    ok=True; parts=[]
    for geo in ("ring","asym"):
        thr=max(0.015,2*V[geo]); m=g[geo]["median"]
        ok &= m<=thr; parts.append(f"{geo} {m:.4f} vs thr {thr:.4f}")
    print(f"  {'PASS' if ok else 'FAIL'}  {NAME.get(arm,arm):34s} K={K[1:]} eps={eps[3:]}  " + "  ".join(parts))
