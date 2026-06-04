"""
Ladder rung 6: RECIPE FIX (hinge anti-identity). Rung-5 diagnosis via the ORACLE_LATENT
control: even with a PERFECT clean-circle latent (enc->z err 0.011), the discovery recipe left
M at identity (angle ~0, recall 0). Cause: the bounded-move term (move-target)^2 has ZERO
gradient at the identity, so M cannot escape it; closure also pulls back toward identity. Fix:
replace with a hinge relu(target-move) (constant nonzero push off identity). Everything else
identical to rung 5. Gate: ORACLE_LATENT must now recover M_angle ~ +-45deg and fill held-out
(recipe validated) BEFORE judging DISC_JOINT (the unsupervised treatment).

Ladder rung 5 (original header): can the LATENT be shaped so the hidden symmetry becomes linear?

Rung 4 result: a reconstruction-faithful frozen latent is a DISTORTED loop, so the 45deg
rotation stays nonlinear there and M collapses to identity. Rung 5 attacks the latent geometry.

KEY NEW CONTROL (decides if 5A is worth anything):
  ORACLE_LATENT : encoder SUPERVISED to the true latent z (clean circle), frozen, dec = true
                  world decoder. Then run the SAME discovery recipe to learn M. Tests:
                  "GIVEN a clean latent, does linear-M discovery fill held-out + recover ~45deg?"
                  Uses privileged z -> it is an UPPER BOUND / control, not a discovery claim.

TREATMENT (5A):
  DISC_JOINT    : enc/dec/M/field trained so the latent is shaped BY the symmetry (unsupervised).
                  --joint-mode {scratch, warm, alt} = three anti-collapse training strategies.
                  Low prior only: latent dim k=2, finite-order P=8, bounded move. NO rotation/
                  orthogonal assumption on M.

Controls carried from rung 3/4: BASE (floor), ORACLE (positive), DISC_LINEAR (negative).
Same world/data/split/metrics/seeds. Read DISC_JOINT only inside [floor, ORACLE].
"""
import argparse, json, math, os
import torch, torch.nn as nn

D=16; K_LAT=2; N_MODES=8; HELDOUT=[2,5]; RADIUS=4.0; SIG_LAT=0.08
STEPS=4000; AE_STEPS=3000; BATCH=512; SAMPLE_STEPS=50; N_SAMPLE=2000; SEEDS=[0,1,2]
W_AUG=1.0; W_MOVE=0.5; W_CYC=1.0; W_RECON=1.0; P_CYCLE=8; W_WD=1e-4; DEC_SEED=1234
ARMS=["BASE","ORACLE","DISC_LINEAR","ORACLE_LATENT","DISC_JOINT"]
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_MODES=[i for i in range(N_MODES) if i not in HELDOUT]

def mlp(sizes, act=nn.SiLU):
    L=[]
    for i in range(len(sizes)-2): L+=[nn.Linear(sizes[i],sizes[i+1]),act()]
    L+=[nn.Linear(sizes[-2],sizes[-1])]; return nn.Sequential(*L)

def build_world():
    g=torch.Generator(device="cpu").manual_seed(DEC_SEED)
    dec=nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,D))
    with torch.no_grad():
        for p in dec.parameters(): p.copy_(torch.randn(p.shape,generator=g)*(1.0 if p.dim()==1 else 0.9))
    dec=dec.to(DEV)
    for p in dec.parameters(): p.requires_grad_(False)
    ang=torch.arange(N_MODES,device=DEV)*(2*math.pi/N_MODES)
    centers=torch.stack([RADIUS*torch.cos(ang),RADIUS*torch.sin(ang)],1)
    with torch.no_grad():
        zc=centers[torch.randint(0,N_MODES,(20000,),device=DEV)]+SIG_LAT*torch.randn(20000,2,device=DEV)
        raw=dec(zc); mu,sd=raw.mean(0),raw.std(0)+1e-6
    decode=lambda z:(dec(z)-mu)/sd
    with torch.no_grad():
        M_obs=decode(centers)
        zc=centers[2].repeat(4000,1)+SIG_LAT*torch.randn(4000,2,device=DEV)
        sig_obs=(decode(zc)-M_obs[2]).norm(dim=1).mean().item()
        pair=torch.cdist(M_obs,M_obs)+torch.eye(N_MODES,device=DEV)*1e9
        min_pair=pair.min().item()
    return decode,centers,M_obs,sig_obs,min_pair

def rot_true():
    t=2*math.pi/N_MODES; c,s=math.cos(t),math.sin(t)
    return torch.tensor([[c,-s],[s,c]],dtype=torch.float32,device=DEV)

def eqm_loss(f,x):
    n=x.shape[0]; g=torch.rand(n,1,device=DEV); eps=torch.randn(n,D,device=DEV)
    xg=(1-g)*x+g*eps
    return ((f(torch.cat([xg,g],1))-(eps-x))**2).mean()

@torch.no_grad()
def sample(f,n):
    x=torch.randn(n,D,device=DEV)
    for i in range(SAMPLE_STEPS):
        g=torch.full((n,1),1-i/SAMPLE_STEPS,device=DEV)
        x=x-f(torch.cat([x,g],1))*(1.0/SAMPLE_STEPS)
    return x

class LinearT(nn.Module):
    def __init__(self):
        super().__init__(); self.A=nn.Parameter(torch.eye(D)+0.05*torch.randn(D,D)); self.b=nn.Parameter(0.05*torch.randn(D))
    def forward(self,x): return x@self.A.T+self.b

class LatentT(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=mlp([D,64,K_LAT]); self.dec=mlp([K_LAT,64,D])
        self.M=nn.Parameter(torch.eye(K_LAT)+0.15*torch.randn(K_LAT,K_LAT))
    def recon(self,x): return self.dec(self.enc(x))
    def forward(self,x): return self.dec(self.enc(x)@self.M.T)

class OracleLatentT(nn.Module):           # enc supervised->z (frozen); dec = true world decoder
    def __init__(self, decode_fn):
        super().__init__(); self.enc=mlp([D,64,K_LAT]); self._decode=decode_fn
        self.M=nn.Parameter(torch.eye(K_LAT)+0.15*torch.randn(K_LAT,K_LAT))
    def forward(self,x): return self._decode(self.enc(x)@self.M.T)

DECODE,CENTERS,M_OBS,SIG_OBS,MIN_PAIR=build_world()
R_TRUE=rot_true()

def sample_latent_z(n):
    idx=torch.tensor(TRAIN_MODES,device=DEV)[torch.randint(0,len(TRAIN_MODES),(n,),device=DEV)]
    return CENTERS[idx]+SIG_LAT*torch.randn(n,2,device=DEV)
def mode_of(x): return torch.cdist(x,M_OBS).argmin(1)

def disc_recipe(loss, f, T, x):
    Tx=T(x)
    loss=loss+W_AUG*eqm_loss(f,Tx)
    move=(Tx-x).pow(2).sum(1).clamp_min(1e-8).sqrt()
    # rung-6 FIX: hinge (linear) anti-identity, NOT squared. (move-target)^2 has ZERO gradient
    # at the identity (move=0) -> M cannot escape identity. relu(target-move) gives a constant
    # nonzero push off identity until move>=target, breaking the identity lock (rung-5 diagnosis:
    # ORACLE_LATENT stuck at M=I even with a perfect clean latent).
    loss=loss+W_MOVE*torch.relu(MIN_PAIR-move).mean()
    xk=x
    for _ in range(P_CYCLE): xk=T(xk)
    loss=loss+W_CYC*(xk-x).pow(2).sum(1).mean()
    return loss

def train(arm, seed, steps, ae_steps, joint_mode):
    torch.manual_seed(seed)
    if DEV.type=="cuda": torch.cuda.manual_seed_all(seed)
    f=mlp([D+1,256,256,D]).to(DEV)
    T=None; recon0=None

    if arm=="DISC_LINEAR":
        T=LinearT().to(DEV)
        opt=torch.optim.Adam(list(f.parameters())+list(T.parameters()),lr=2e-3,weight_decay=W_WD)
    elif arm=="ORACLE_LATENT":
        T=OracleLatentT(DECODE).to(DEV)
        # pretrain enc SUPERVISED to true z, then freeze enc
        eopt=torch.optim.Adam(T.enc.parameters(),lr=2e-3)
        for _ in range(ae_steps):
            z=sample_latent_z(BATCH); x=DECODE(z); eopt.zero_grad()
            ((T.enc(x)-z)**2).mean().backward(); eopt.step()
        with torch.no_grad():
            z=sample_latent_z(2048); x=DECODE(z)
            recon0=float((T.enc(x)-z).norm(dim=1).mean()/z.norm(dim=1).mean())  # enc->z error
        for p in T.enc.parameters(): p.requires_grad_(False)
        opt=torch.optim.Adam(list(f.parameters())+[T.M],lr=2e-3)
    elif arm=="DISC_JOINT":
        T=LatentT().to(DEV)
        if joint_mode=="warm":
            aopt=torch.optim.Adam(list(T.enc.parameters())+list(T.dec.parameters()),lr=2e-3)
            for _ in range(ae_steps):
                x=DECODE(sample_latent_z(BATCH)); aopt.zero_grad(); ((T.recon(x)-x)**2).mean().backward(); aopt.step()
        opt=torch.optim.Adam(list(f.parameters())+list(T.parameters()),lr=2e-3,weight_decay=W_WD)
    else:
        opt=torch.optim.Adam(f.parameters(),lr=2e-3)

    for step in range(steps):
        z=sample_latent_z(BATCH); x=DECODE(z)
        if arm=="DISC_JOINT" and joint_mode=="alt" and step%2==0:
            # alternating: even step = recon-only update (structurally protect the AE)
            opt.zero_grad(); (W_RECON*((T.recon(x)-x)**2).mean()).backward(); opt.step(); continue
        opt.zero_grad()
        loss=eqm_loss(f,x)
        if arm=="ORACLE":
            loss=loss+W_AUG*eqm_loss(f,DECODE(z@R_TRUE.T))
        elif arm in ("DISC_LINEAR","ORACLE_LATENT"):
            loss=disc_recipe(loss,f,T,x)
        elif arm=="DISC_JOINT":
            loss=disc_recipe(loss,f,T,x)+W_RECON*((T.recon(x)-x)**2).mean()
        loss.backward(); opt.step()

    # final recon diagnostic for joint
    if arm=="DISC_JOINT":
        with torch.no_grad():
            x=DECODE(sample_latent_z(2048)); recon0=float((T.recon(x)-x).norm(dim=1).mean()/x.norm(dim=1).mean())
    return f,T,recon0

def energy_distance(a,b):
    return (2*torch.cdist(a,b).mean()-torch.cdist(a,a).mean()-torch.cdist(b,b).mean()).item()

@torch.no_grad()
def evaluate(f,T,recon0):
    s=sample(f,N_SAMPLE); d=torch.cdist(s,M_OBS); near=d.argmin(1); within=d.min(1).values<3*SIG_OBS
    frac=torch.zeros(N_MODES,device=DEV)
    for m in range(N_MODES): frac[m]=((near==m)&within).float().sum()
    frac=frac/N_SAMPLE
    ref=DECODE(CENTERS[torch.randint(0,N_MODES,(N_SAMPLE,),device=DEV)]+SIG_LAT*torch.randn(N_SAMPLE,2,device=DEV))
    out={"recall_heldout":float(frac[HELDOUT].sum()),"modes_covered":int((frac>0.01).sum()),
         "mmd_energy":energy_distance(s,ref)}
    if T is not None:
        xb=DECODE(sample_latent_z(512)); Tx=T(xb)
        out["T_on_manifold"]=float((torch.cdist(Tx,M_OBS).min(1).values<3*SIG_OBS).float().mean())
        sigma={}
        for i in TRAIN_MODES:
            xi=DECODE(CENTERS[i].repeat(128,1)+SIG_LAT*torch.randn(128,2,device=DEV))
            sigma[i]=int(torch.bincount(mode_of(T(xi)),minlength=N_MODES).argmax().item())
        out["sigma"]=sigma; out["hits_heldout"]=bool(any(v in HELDOUT for v in sigma.values()))
        out["consistent_shift"]=(len({(sigma[i]-i)%N_MODES for i in TRAIN_MODES})==1)
        if hasattr(T,"M"):
            I=torch.eye(K_LAT,device=DEV); Mp=T.M.clone()
            for _ in range(P_CYCLE-1): Mp=Mp@T.M
            out["M_offidentity"]=float((T.M-I).norm()); out["M_order_err"]=float((Mp-I).norm())
            out["M_angle_deg"]=float(torch.atan2(T.M[1,0],T.M[0,0])*180/math.pi)
    if recon0 is not None: out["ae_diag"]=recon0
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-json",default="results_rung5.json")
    ap.add_argument("--joint-mode",choices=["scratch","warm","alt"],default="scratch")
    ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()
    steps,ae_steps,seeds=(300,300,[0]) if args.quick else (STEPS,AE_STEPS,SEEDS)
    print(f"device={DEV} K_lat={K_LAT} heldout={HELDOUT} joint_mode={args.joint_mode}")
    print(f"decoder ratio min_pair/sig_obs={MIN_PAIR/SIG_OBS:.1f}  ideal recall={len(HELDOUT)/N_MODES:.3f}\n")
    results={"config":{"joint_mode":args.joint_mode,"steps":steps,"seeds":seeds,"device":str(DEV)},"arms":{}}
    hdr=f"{'arm':14s} {'recall@HO':>10s} {'modes/8':>8s} {'T_onman':>8s} {'cyc':>5s} {'AE/enc':>7s} {'M-I':>6s} {'M^8-I':>7s} {'Mang':>7s}"
    print(hdr); print("-"*len(hdr))
    for arm in ARMS:
        runs=[evaluate(*train(arm,s,steps,ae_steps,args.joint_mode)) for s in seeds]
        agg={}
        for k in ("recall_heldout","modes_covered","mmd_energy","T_on_manifold","ae_diag","M_offidentity","M_order_err","M_angle_deg"):
            v=[r[k] for r in runs if k in r]
            if v: agg[k]=sum(v)/len(v)
        agg["recall_sd"]=(sum((r["recall_heldout"]-agg["recall_heldout"])**2 for r in runs)/len(runs))**0.5
        agg["consistent_shift_any"]=any(r.get("consistent_shift") for r in runs)
        agg["sigma_seed0"]=runs[0].get("sigma")
        results["arms"][arm]=agg
        def gg(k,fmt="%.3f",w=7): return (fmt%agg[k]).rjust(w) if k in agg else "   -  ".rjust(w)
        print(f"{arm:14s} {agg['recall_heldout']:6.3f}±{agg['recall_sd']:.3f} {agg['modes_covered']:8.1f} "
              f"{gg('T_on_manifold',w=8)} {str(agg['consistent_shift_any']):>5s} {gg('ae_diag')} "
              f"{gg('M_offidentity','%.2f',6)} {gg('M_order_err','%.3f',7)} {gg('M_angle_deg','%.1f',7)}")
    print(f"\nideal recall@HO={len(HELDOUT)/N_MODES:.3f}  (M_angle ~45 or multiple => real rotation found)")
    print("sigma(seed0) ORACLE_LATENT:",results["arms"]["ORACLE_LATENT"].get("sigma_seed0"))
    print("sigma(seed0) DISC_JOINT   :",results["arms"]["DISC_JOINT"].get("sigma_seed0"))
    os.makedirs(os.path.dirname(args.out_json) or ".",exist_ok=True)
    json.dump(results,open(args.out_json,"w"),indent=2)
    print(f"\nwrote {args.out_json}")

if __name__=="__main__": main()
