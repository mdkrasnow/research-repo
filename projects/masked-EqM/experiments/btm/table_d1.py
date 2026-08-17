import json,glob,statistics as st
ARM={"btm_IIB_V_s0":"V   vector BTM","btm_IIB_G_s0":"G   scalar exact",
 "btm_IIB_D1_s0":"D1  FD directional K=1","btm_IIB_D4_s0":"D4  FD directional K=4",
 "btm_IIB_LEGACYvec_s0":"LEGACY vector (neg)","btm_IIB_LEGACYscalar_s0":"LEGACY direct scalar (neg)"}
order=list(ARM)
E={}
for d in sorted(glob.glob("*/*/gradient_metrics.jsonl")):
    tag=d.split("/")[0].rsplit("_job",1)[0]
    E[tag]=[json.loads(l) for l in open(d) if '"target_cosine"' in l]

print("TABLE D.1 -- BTM target matching at eval (Phase II-B, ImageNet B/2, tc=0.9)")
print("target_cosine = cos(model field, corrected BTM population target b*). Eval-only;")
print("never in the training graph. 'near_data' / 'far' split the interpolant by gamma.\n")
h=(f"{'arm':28s} {'n':>4s} {'maxstep':>8s} | {'cos ALL':>8s} {'cos near':>9s} {'cos far':>8s} "
   f"| {'mse/dim':>8s} | {'|f| ratio':>9s} | {'E mean':>9s} {'E std':>8s}")
print(h); print("-"*len(h))
def med(rs,k):
    v=[r[k] for r in rs if r.get(k) is not None]
    return st.median(v) if v else float("nan")
for tag in order:
    rs=E.get(tag) or []
    if not rs: continue
    # final 20% window: the converged value, not the average over the ramp
    rs=rs[int(len(rs)*0.8):]
    print(f"{ARM[tag]:28s} {len(E[tag]):4d} {max(r['step'] for r in E[tag]):8d} | "
          f"{med(rs,'target_cosine'):8.4f} {med(rs,'target_cosine_near_data'):9.4f} "
          f"{med(rs,'target_cosine_far'):8.4f} | {med(rs,'target_mse_per_dim'):8.4f} | "
          f"{med(rs,'target_norm_ratio'):9.4f} | {med(rs,'E_mean'):+9.3f} {med(rs,'E_std'):8.3f}")

print("\nSTEP-MATCHED (final 20% of the window [0, S] for each S):")
for S in (11700,22800,66900,75000):
    print(f"--- through step {S} ---")
    for tag in order:
        rs=[r for r in (E.get(tag) or []) if r["step"]<=S]
        if len(rs)<5: continue
        w=rs[int(len(rs)*0.8):]
        print(f"  {ARM[tag]:28s} n={len(rs):4d}  cos {med(w,'target_cosine'):.4f}  "
              f"near {med(w,'target_cosine_near_data'):.4f}  far {med(w,'target_cosine_far'):.4f}  "
              f"mse/dim {med(w,'target_mse_per_dim'):.4f}")
    print()
