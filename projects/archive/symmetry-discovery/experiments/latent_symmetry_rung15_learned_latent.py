"""
Ladder rung 15: LEARNED latent (remove the supervised enc->z idealization).

Pre-register:
  Question: does the stable-generator recipe still reach near-oracle when the latent is NOT supervised
            to true z (the biggest remaining toy idealization)?
  World: rung-14 K=3 cylinder (z=[r cosφ, r sinφ, h]; symmetry φ→φ+Δ, h unchanged). Held-out φ-arc.
  Operator (fixed across arms): stable group generator M=exp(A), reg = det≈1 + cond→1 (NO skew/plane).
  Vary ONLY the latent the operator lives in:
    GEN_FREE_STABLE_SUPERVISED_LATENT  enc supervised->true z, dec=true decoder (rung-14 reference upper)
    GEN_FREE_STABLE_RECON_LATENT_FROZEN  AE (enc/dec) recon-pretrained then FROZEN; M in that latent
    GEN_FREE_STABLE_LEARNED_LATENT       AE recon-pretrained, then enc/dec + M co-trained vs frozen
                                          anchor with a RECON FLOOR (latent may reshape iff recon stays good)
  Controls: BASE (floor), ORACLE (true-symmetry aug, positive).
  Frozen anchor = energy distance to UNLABELED full-manifold samples (observed space). NO live EqM
  closure. NO held-out EqM targets.
  Success: LEARNED approaches SUPERVISED & ORACLE. Partial: LEARNED beats RECON_FROZEN/floor with good
  recon + coherent shifts. Failure: LEARNED == floor despite good recon, or only works by wrecking recon.
  Kill: if learned-latent fails twice under good recon, stop retuning; report negative.

Key coordinate-free diagnostic: passive_dh (true h preserved, via observed->true-(φ,h) grid) works for
ALL arms regardless of latent coords. M-based passive_leak only meaningful for the supervised arm.
"""
import argparse, json, math, os
import torch, torch.nn as nn

D=16; K_LAT=3; RAD=4.0; HRANGE=1.5; SIG=0.08
GAP_LO=157.5*math.pi/180; GAP_HI=202.5*math.pi/180
T_STEPS=3000; AE_STEPS=3000; EQM_STEPS=4000; BATCH=512; SAMPLE_STEPS=50; N_SAMPLE=3000; SEEDS=[0,1,2]
ALPHA_AUG=1.0; LAM_MOVE=0.5; LAM_DET=1.0; LAM_COND=1.0; W_RECON_FLOOR=5.0; DEC_SEED=1234; TARGET_DEG=60
ARMS=["BASE","ORACLE","GEN_FREE_STABLE_SUPERVISED_LATENT","GEN_FREE_STABLE_RECON_LATENT_FROZEN","GEN_FREE_STABLE_LEARNED_LATENT"]
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mlp(sizes, act=nn.SiLU):
    L=[]
    for i in range(len(sizes)-2): L+=[nn.Linear(sizes[i],sizes[i+1]),act()]
    L+=[nn.Linear(sizes[-2],sizes[-1])]; return nn.Sequential(*L)

def build_world():
    g=torch.Generator(device="cpu").manual_seed(DEC_SEED)
    dec=nn.Sequential(nn.Linear(K_LAT,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,D))
    with torch.no_grad():
        for p in dec.parameters(): p.copy_(torch.randn(p.shape,generator=g)*(1.0 if p.dim()==1 else 0.9))
    dec=dec.to(DEV)
    for p in dec.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        phi=torch.rand(40000,device=DEV)*2*math.pi; h=(torch.rand(40000,device=DEV)*2-1)*HRANGE
        z=torch.stack([RAD*torch.cos(phi),RAD*torch.sin(phi),h],1)+SIG*torch.randn(40000,3,device=DEV)
        raw=dec(z); mu,sd=raw.mean(0),raw.std(0)+1e-6
    decode=lambda z:(dec(z)-mu)/sd
    with torch.no_grad():
        pg=torch.linspace(0,2*math.pi,73,device=DEV)[:-1]; hg=torch.linspace(-HRANGE,HRANGE,7,device=DEV)
        PP,HH=torch.meshgrid(pg,hg,indexing="ij"); PP=PP.reshape(-1); HH=HH.reshape(-1)
        man=decode(torch.stack([RAD*torch.cos(PP),RAD*torch.sin(PP),HH],1))
        z0=torch.stack([RAD*torch.cos(torch.zeros(4000,device=DEV)),RAD*torch.sin(torch.zeros(4000,device=DEV)),torch.zeros(4000,device=DEV)],1)+SIG*torch.randn(4000,3,device=DEV)
        sig_obs=(decode(z0)-man[0]).norm(dim=1).mean().item()
    return decode,man,PP,HH,sig_obs

DECODE,MAN,GRID_PHI,GRID_H,SIG_OBS=build_world()
TARGET_MOVE=float((DECODE(torch.tensor([[RAD*math.cos(math.radians(TARGET_DEG)),RAD*math.sin(math.radians(TARGET_DEG)),0.0]],device=DEV))
                   -DECODE(torch.tensor([[RAD,0.0,0.0]],device=DEV))).norm())

def in_gap(phi): return (phi>=GAP_LO)&(phi<=GAP_HI)
def z_of(phi,h): return torch.stack([RAD*torch.cos(phi),RAD*torch.sin(phi),h],1)
def sample_h(n): return (torch.rand(n,device=DEV)*2-1)*HRANGE
def sample_visible(n):
    out=torch.empty(0,device=DEV)
    while out.numel()<n:
        p=torch.rand(n,device=DEV)*2*math.pi; p=p[~in_gap(p)]; out=torch.cat([out,p])
    phi=out[:n]; return DECODE(z_of(phi,sample_h(n))+SIG*torch.randn(n,3,device=DEV))
def sample_full(n):
    phi=torch.rand(n,device=DEV)*2*math.pi; return DECODE(z_of(phi,sample_h(n))+SIG*torch.randn(n,3,device=DEV))
def eqm_loss(f,x):
    n=x.shape[0]; g=torch.rand(n,1,device=DEV); eps=torch.randn(n,D,device=DEV); xg=(1-g)*x+g*eps
    return ((f(torch.cat([xg,g],1))-(eps-x))**2).mean()
def energy_distance(a,b):
    return 2*torch.cdist(a,b).mean()-torch.cdist(a,a).mean()-torch.cdist(b,b).mean()
@torch.no_grad()
def sample(f,n):
    x=torch.randn(n,D,device=DEV)
    for i in range(SAMPLE_STEPS):
        g=torch.full((n,1),1-i/SAMPLE_STEPS,device=DEV); x=x-f(torch.cat([x,g],1))*(1.0/SAMPLE_STEPS)
    return x

class StableGenOp(nn.Module):
    """Stable group generator M=exp(A) acting in a latent. learned_dec=False uses true DECODE (supervised
       latent); True uses a learned decoder (recon/learned latent)."""
    def __init__(self, learned_dec):
        super().__init__(); self.enc=mlp([D,64,K_LAT]); self.learned=learned_dec
        if learned_dec: self.dec=mlp([K_LAT,64,D])
        self.A=nn.Parameter(0.10*torch.randn(K_LAT,K_LAT))
    def decode(self,z): return self.dec(z) if self.learned else DECODE(z)
    def M(self): return torch.matrix_exp(self.A)
    def recon(self,x): return self.decode(self.enc(x))
    def reg(self):
        M=self.M(); det=torch.det(M); sv=torch.linalg.svdvals(M)
        cond=(sv.max()/sv.min().clamp_min(1e-6)-1.0)**2
        return LAM_DET*(torch.log(det.abs()+1e-8))**2 + LAM_COND*cond
    def forward(self,x): return self.decode(self.enc(x)@self.M().T)

def pretrain_latent(T, mode, seed):
    torch.manual_seed(seed)
    if mode=="sup":
        opt=torch.optim.Adam(T.enc.parameters(),lr=2e-3)
        for _ in range(AE_STEPS):
            phi=torch.rand(BATCH,device=DEV)*2*math.pi; z=z_of(phi,sample_h(BATCH)); x=DECODE(z+SIG*torch.randn_like(z))
            opt.zero_grad(); ((T.enc(x)-z)**2).mean().backward(); opt.step()
        with torch.no_grad():
            phi=torch.rand(2048,device=DEV)*2*math.pi; z=z_of(phi,sample_h(2048)); x=DECODE(z+SIG*torch.randn_like(z))
            recon0=float((T.enc(x)-z).norm(dim=1).mean()/z.norm(dim=1).mean())
    else:  # recon AE
        opt=torch.optim.Adam(list(T.enc.parameters())+list(T.dec.parameters()),lr=2e-3)
        for _ in range(AE_STEPS):
            x=sample_full(BATCH); opt.zero_grad(); ((T.recon(x)-x)**2).mean().backward(); opt.step()
        with torch.no_grad():
            x=sample_full(2048); recon0=float((T.recon(x)-x).norm(dim=1).mean()/x.norm(dim=1).mean())
    return recon0

def recon_metric(T, mode):
    with torch.no_grad():
        if mode=="sup":
            phi=torch.rand(2048,device=DEV)*2*math.pi; z=z_of(phi,sample_h(2048)); x=DECODE(z+SIG*torch.randn_like(z))
            return float((T.enc(x)-z).norm(dim=1).mean()/z.norm(dim=1).mean())
        x=sample_full(2048); return float((T.recon(x)-x).norm(dim=1).mean()/x.norm(dim=1).mean())

def discover_operator(seed, mode):
    """mode in {sup, recon_frozen, learned}. Returns (T, recon_pre, recon_final)."""
    torch.manual_seed(seed); learned_dec=(mode!="sup"); T=StableGenOp(learned_dec).to(DEV)
    recon_pre=pretrain_latent(T, "sup" if mode=="sup" else "recon", seed)
    if mode in ("sup","recon_frozen"):
        for p in T.enc.parameters(): p.requires_grad_(False)
        if learned_dec:
            for p in T.dec.parameters(): p.requires_grad_(False)
        opt=torch.optim.Adam([T.A],lr=2e-3)
        for _ in range(T_STEPS):
            xa=sample_full(BATCH); xb=sample_full(BATCH); tx=T(xa); opt.zero_grad()
            move=(tx-xa).norm(dim=1).mean()
            (energy_distance(tx,xb)+LAM_MOVE*(move-TARGET_MOVE)**2+T.reg()).backward(); opt.step()
        for p in [T.A]: p.requires_grad_(False)
    else:  # learned: co-train enc/dec/A vs frozen anchor + RECON FLOOR
        opt=torch.optim.Adam(list(T.enc.parameters())+list(T.dec.parameters())+[T.A],lr=2e-3)
        for _ in range(T_STEPS):
            xa=sample_full(BATCH); xb=sample_full(BATCH); opt.zero_grad()
            tx=T(xa); move=(tx-xa).norm(dim=1).mean()
            loss=energy_distance(tx,xb)+LAM_MOVE*(move-TARGET_MOVE)**2+T.reg()
            loss=loss+W_RECON_FLOOR*((T.recon(xa)-xa)**2).mean()        # recon floor: latent must stay faithful
            loss.backward(); opt.step()
        for p in list(T.enc.parameters())+list(T.dec.parameters())+[T.A]: p.requires_grad_(False)
    return T, recon_pre, recon_metric(T, "sup" if mode=="sup" else "recon")

def train_field(seed, T=None, oracle=False):
    torch.manual_seed(seed); f=mlp([D+1,256,256,D]).to(DEV); opt=torch.optim.Adam(f.parameters(),lr=2e-3)
    for _ in range(EQM_STEPS):
        x=sample_visible(BATCH); opt.zero_grad(); loss=eqm_loss(f,x)
        if oracle:
            th=float(torch.rand(1))*2*math.pi; phi=torch.rand(BATCH,device=DEV)*2*math.pi; phi=phi[~in_gap(phi)]
            loss=loss+ALPHA_AUG*eqm_loss(f,DECODE(z_of(phi+th, sample_h(phi.shape[0]))))
        elif T is not None:
            loss=loss+ALPHA_AUG*eqm_loss(f,T(x).detach())
        loss.backward(); opt.step()
    return f

def run_arm(arm, seed):
    if arm=="BASE": return train_field(seed), None, None, None
    if arm=="ORACLE": return train_field(seed, oracle=True), None, None, None
    mode={"GEN_FREE_STABLE_SUPERVISED_LATENT":"sup","GEN_FREE_STABLE_RECON_LATENT_FROZEN":"recon_frozen","GEN_FREE_STABLE_LEARNED_LATENT":"learned"}[arm]
    T,rp,rf=discover_operator(seed,mode); return train_field(seed,T=T), T, rp, rf

@torch.no_grad()
def nearest(x):
    d=torch.cdist(x,MAN); idx=d.argmin(1); return GRID_PHI[idx], GRID_H[idx], d.min(1).values

@torch.no_grad()
def evaluate(f,T,recon_pre,recon_final):
    s=sample(f,N_SAMPLE); phi,h,dist=nearest(s); within=dist<3*SIG_OBS
    rec=float((within&in_gap(phi)).float().sum()/N_SAMPLE)
    bins=torch.floor(phi/(2*math.pi)*36).clamp(0,35).long()
    cov=torch.tensor([((bins==b)&within).float().sum() for b in range(36)],device=DEV)/N_SAMPLE
    out={"recall_arc":rec,"bins":int((cov>0.005).sum()),"mmd_full":float(energy_distance(s,sample_full(N_SAMPLE)))}
    if T is not None:
        xa=sample_full(1024); Tx=T(xa)
        p0,h0,_=nearest(xa); p1,h1,d1=nearest(Tx)
        out["T_onman"]=float((d1<3*SIG_OBS).float().mean())
        dphi=((p1-p0+math.pi)%(2*math.pi))-math.pi
        out["T_shift_std"]=float(dphi.std()*180/math.pi)
        out["passive_dh"]=float((h1-h0).abs().mean())
        M=T.M(); sv=torch.linalg.svdvals(M)
        out["M_det"]=float(torch.det(M)); out["M_cond"]=float(sv.max()/sv.min().clamp_min(1e-6))
        out["recon_pre"]=recon_pre; out["recon_final"]=recon_final
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-json",default="results_rung15.json"); ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()
    global T_STEPS,AE_STEPS,EQM_STEPS
    seeds=SEEDS
    if args.quick: T_STEPS,AE_STEPS,EQM_STEPS,seeds=400,400,400,[0]
    print(f"device={DEV} K_LAT={K_LAT} TARGET_MOVE={TARGET_MOVE:.3f} ideal recall_arc={45/360:.3f}\n")
    results={"config":{"K_LAT":K_LAT,"seeds":seeds,"device":str(DEV)},"arms":{}}
    hdr=f"{'arm':38s} {'recall':>7s} {'%ORC':>5s} {'bins':>4s} {'shiftSD':>7s} {'pass_dh':>7s} {'det':>5s} {'cond':>5s} {'rec_pre':>7s} {'rec_fin':>7s}"
    print(hdr); print("-"*len(hdr))
    orc=None
    for arm in ARMS:
        runs=[evaluate(*run_arm(arm,s)) for s in seeds]
        agg={}
        for k in ("recall_arc","bins","T_onman","T_shift_std","passive_dh","M_det","M_cond","recon_pre","recon_final"):
            v=[r[k] for r in runs if k in r]
            if v: agg[k]=sum(v)/len(v)
        agg["recall_sd"]=(sum((r["recall_arc"]-agg["recall_arc"])**2 for r in runs)/len(runs))**0.5
        results["arms"][arm]=agg
        if arm=="ORACLE": orc=agg["recall_arc"]
        pct=(100*agg["recall_arc"]/orc) if orc else 0
        def gg(k,fmt="%.3f",w=5): return (fmt%agg[k]).rjust(w) if k in agg else "  -  ".rjust(w)
        print(f"{arm:38s} {agg['recall_arc']:6.3f} {pct:5.0f} {agg['bins']:4.0f} {gg('T_shift_std','%.1f',7)} "
              f"{gg('passive_dh','%.3f',7)} {gg('M_det',w=5)} {gg('M_cond',w=5)} {gg('recon_pre','%.3f',7)} {gg('recon_final','%.3f',7)}")
    print(f"\nideal recall_arc={45/360:.3f}. KEY: does LEARNED_LATENT approach SUPERVISED_LATENT & ORACLE w/ good recon?")
    os.makedirs(os.path.dirname(args.out_json) or ".",exist_ok=True)
    json.dump(results,open(args.out_json,"w"),indent=2); print(f"wrote {args.out_json}")

if __name__=="__main__": main()
