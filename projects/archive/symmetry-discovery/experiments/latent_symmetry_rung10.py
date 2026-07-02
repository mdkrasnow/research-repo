"""
Ladder rung 10: frozen anchor + COHERENT operator (single latent matrix).

Rung 9 result: a frozen data anchor (energy-distance to unlabeled full-manifold samples) FIXES field
co-adaptation -> the operator stays on-manifold (T_onman 0.70). But a free residual-MLP operator under
distributional anchoring becomes a RANDOM on-manifold shuffle (shift_std ~95deg), not a coherent flow,
so it does not fill the held-out arc.

Rung 10 = same frozen anchor, ONE mechanism change: make the operator coherent BY CONSTRUCTION as a
single latent linear map  T(x) = dec(M . enc(x)),  M a 2x2 applied to EVERY point -> one consistent
shift. If this fills the arc with low shift_std, operator-coherence was the missing piece.

Arms:
  BASE                  EqM visible only                                  (floor)
  ORACLE                EqM + true-group (random SO(2)) aug               (positive control)
  FROZEN_RESIDUAL       rung-9 frozen-anchor residual-MLP operator        (reproduce incoherence)
  FROZEN_LATENT_CLEAN   frozen anchor + single M, enc SUPERVISED->z       (isolate coherence; clean latent)
  FROZEN_LATENT_DISC    frozen anchor + single M, enc from recon AE       (fully unsupervised)

All M/T trained against the FROZEN anchor BEFORE EqM, then frozen, then used as EqM augmentation.
WIN = FROZEN_LATENT_* fills arc (recall up, shift_std LOW, M_angle ~ a real rotation) while
FROZEN_RESIDUAL stays incoherent.
"""
import argparse, json, math, os
import torch, torch.nn as nn

D=16; K_LAT=2; RAD=4.0; SIG=0.08
GAP_LO=157.5*math.pi/180; GAP_HI=202.5*math.pi/180
T_STEPS=3000; AE_STEPS=3000; EQM_STEPS=4000; BATCH=512; SAMPLE_STEPS=50; N_SAMPLE=3000; SEEDS=[0,1,2]
ALPHA_AUG=1.0; LAM_MOVE=0.5; W_WD=1e-4; DEC_SEED=1234; T_RES_SCALE=0.3; TARGET_DEG=60
ARMS=["BASE","ORACLE","FROZEN_RESIDUAL","FROZEN_LATENT_CLEAN","FROZEN_LATENT_DISC"]
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    with torch.no_grad():
        phi=torch.rand(20000,device=DEV)*2*math.pi
        z=torch.stack([RAD*torch.cos(phi),RAD*torch.sin(phi)],1)+SIG*torch.randn(20000,2,device=DEV)
        raw=dec(z); mu,sd=raw.mean(0),raw.std(0)+1e-6
    decode=lambda z:(dec(z)-mu)/sd
    with torch.no_grad():
        pg=torch.linspace(0,2*math.pi,361,device=DEV)[:-1]
        man=decode(torch.stack([RAD*torch.cos(pg),RAD*torch.sin(pg)],1))
        z0=torch.stack([RAD*torch.cos(torch.zeros(4000,device=DEV)),RAD*torch.sin(torch.zeros(4000,device=DEV))],1)+SIG*torch.randn(4000,2,device=DEV)
        sig_obs=(decode(z0)-man[0]).norm(dim=1).mean().item()
    return decode,pg,man,sig_obs

DECODE,PHI_GRID,MAN,SIG_OBS=build_world()
TARGET_MOVE=float((MAN[TARGET_DEG]-MAN[0]).norm())

def in_gap(phi): return (phi>=GAP_LO)&(phi<=GAP_HI)
def z_of(phi): return torch.stack([RAD*torch.cos(phi),RAD*torch.sin(phi)],1)
def sample_visible(n):
    out=torch.empty(0,device=DEV)
    while out.numel()<n:
        p=torch.rand(n,device=DEV)*2*math.pi; p=p[~in_gap(p)]; out=torch.cat([out,p])
    return DECODE(z_of(out[:n])+SIG*torch.randn(n,2,device=DEV))
def sample_full(n):
    phi=torch.rand(n,device=DEV)*2*math.pi
    return DECODE(z_of(phi)+SIG*torch.randn(n,2,device=DEV))
def rot_mat(t):
    c,s=math.cos(t),math.sin(t); return torch.tensor([[c,-s],[s,c]],dtype=torch.float32,device=DEV)
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

class ResidualT(nn.Module):
    def __init__(self):
        super().__init__(); self.h=mlp([D,128,128,D]); self.s=T_RES_SCALE
    def forward(self,x): return x+self.s*self.h(x)
class LatentLinearT(nn.Module):
    """Coherent operator: ONE matrix M applied to all encoded points. dec is DECODE (clean) or learned."""
    def __init__(self, decode_fn, learned_dec=False):
        super().__init__(); self.enc=mlp([D,64,K_LAT]); self.M=nn.Parameter(torch.eye(K_LAT)+0.15*torch.randn(K_LAT,K_LAT))
        self.learned=learned_dec
        if learned_dec: self.dec=mlp([K_LAT,64,D])
        self._decode=decode_fn
    def decode(self,z): return self.dec(z) if self.learned else self._decode(z)
    def recon(self,x): return self.decode(self.enc(x))
    def forward(self,x): return self.decode(self.enc(x)@self.M.T)

def anchor_train_residual(seed):
    torch.manual_seed(seed); T=ResidualT().to(DEV)
    opt=torch.optim.Adam(T.parameters(),lr=2e-3,weight_decay=W_WD)
    for _ in range(T_STEPS):
        xa=sample_full(BATCH); xb=sample_full(BATCH); tx=T(xa); opt.zero_grad()
        move=(tx-xa).norm(dim=1).mean()
        (energy_distance(tx,xb)+LAM_MOVE*(move-TARGET_MOVE)**2).backward(); opt.step()
    for p in T.parameters(): p.requires_grad_(False)
    return T

def anchor_train_latent(seed, learned_dec):
    torch.manual_seed(seed); T=LatentLinearT(DECODE, learned_dec=learned_dec).to(DEV)
    # 1) build the latent: enc supervised->z (clean) OR enc/dec recon AE (disc); then FREEZE enc(/dec)
    if learned_dec:
        ao=torch.optim.Adam(list(T.enc.parameters())+list(T.dec.parameters()),lr=2e-3)
        for _ in range(AE_STEPS):
            x=sample_full(BATCH); ao.zero_grad(); ((T.recon(x)-x)**2).mean().backward(); ao.step()
        for p in list(T.enc.parameters())+list(T.dec.parameters()): p.requires_grad_(False)
    else:
        eo=torch.optim.Adam(T.enc.parameters(),lr=2e-3)
        for _ in range(AE_STEPS):
            phi=torch.rand(BATCH,device=DEV)*2*math.pi; z=z_of(phi); x=DECODE(z+SIG*torch.randn_like(z))
            eo.zero_grad(); ((T.enc(x)-z)**2).mean().backward(); eo.step()
        for p in T.enc.parameters(): p.requires_grad_(False)
    # 2) learn the single matrix M against the FROZEN anchor
    opt=torch.optim.Adam([T.M],lr=2e-3)
    for _ in range(T_STEPS):
        xa=sample_full(BATCH); xb=sample_full(BATCH); tx=T(xa); opt.zero_grad()
        move=(tx-xa).norm(dim=1).mean()
        (energy_distance(tx,xb)+LAM_MOVE*(move-TARGET_MOVE)**2).backward(); opt.step()
    T.M.requires_grad_(False)
    return T

def train_field(seed, T=None, oracle=False):
    torch.manual_seed(seed); f=mlp([D+1,256,256,D]).to(DEV); opt=torch.optim.Adam(f.parameters(),lr=2e-3)
    for _ in range(EQM_STEPS):
        x=sample_visible(BATCH); opt.zero_grad(); loss=eqm_loss(f,x)
        if oracle:
            th=float(torch.rand(1))*2*math.pi; phi=torch.rand(BATCH,device=DEV)*2*math.pi; phi=phi[~in_gap(phi)]
            loss=loss+ALPHA_AUG*eqm_loss(f,DECODE(z_of(phi)@rot_mat(th).T))
        elif T is not None:
            loss=loss+ALPHA_AUG*eqm_loss(f,T(x).detach())
        loss.backward(); opt.step()
    return f

def run_arm(arm, seed):
    if arm=="BASE": return train_field(seed), None
    if arm=="ORACLE": return train_field(seed, oracle=True), None
    if arm=="FROZEN_RESIDUAL": T=anchor_train_residual(seed); return train_field(seed,T=T), T
    if arm=="FROZEN_LATENT_CLEAN": T=anchor_train_latent(seed, learned_dec=False); return train_field(seed,T=T), T
    if arm=="FROZEN_LATENT_DISC":  T=anchor_train_latent(seed, learned_dec=True);  return train_field(seed,T=T), T

@torch.no_grad()
def evaluate(f,T):
    s=sample(f,N_SAMPLE); d=torch.cdist(s,MAN); near=d.argmin(1); within=d.min(1).values<3*SIG_OBS
    phi=PHI_GRID[near]; rec=float((within&in_gap(phi)).float().sum()/N_SAMPLE)
    bins=torch.floor(phi/(2*math.pi)*36).clamp(0,35).long()
    cov=torch.tensor([((bins==b)&within).float().sum() for b in range(36)],device=DEV)/N_SAMPLE
    out={"recall_arc":rec,"bins":int((cov>0.005).sum()),"mmd_full":float(energy_distance(s,sample_full(N_SAMPLE)))}
    if T is not None:
        xa=sample_full(1024); Tx=T(xa)
        out["T_onman"]=float((torch.cdist(Tx,MAN).min(1).values<3*SIG_OBS).float().mean())
        out["T_move"]=float((Tx-xa).norm(dim=1).mean()/TARGET_MOVE)
        p0=PHI_GRID[torch.cdist(xa,MAN).argmin(1)]; p1=PHI_GRID[torch.cdist(Tx,MAN).argmin(1)]
        dphi=((p1-p0+math.pi)%(2*math.pi))-math.pi
        out["T_shift_deg"]=float(dphi.mean()*180/math.pi); out["T_shift_std"]=float(dphi.std()*180/math.pi)
        if hasattr(T,"M"): out["M_angle"]=float(torch.atan2(T.M[1,0],T.M[0,0])*180/math.pi); out["M_off"]=float((T.M-torch.eye(K_LAT,device=DEV)).norm())
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-json",default="results_rung10.json"); ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()
    global T_STEPS,AE_STEPS,EQM_STEPS
    seeds=SEEDS
    if args.quick: T_STEPS,AE_STEPS,EQM_STEPS,seeds=400,400,400,[0]
    print(f"device={DEV} TARGET_MOVE({TARGET_DEG}deg)={TARGET_MOVE:.3f} ideal recall_arc={45/360:.3f}\n")
    results={"config":{"seeds":seeds,"device":str(DEV)},"arms":{}}
    hdr=f"{'arm':20s} {'recall_arc':>10s} {'bins/36':>8s} {'T_onman':>8s} {'shift°':>7s} {'shiftSD':>8s} {'M_ang':>7s} {'M_off':>6s}"
    print(hdr); print("-"*len(hdr))
    for arm in ARMS:
        runs=[evaluate(*run_arm(arm,s)) for s in seeds]
        agg={}
        for k in ("recall_arc","bins","T_onman","T_shift_deg","T_shift_std","M_angle","M_off"):
            v=[r[k] for r in runs if k in r]
            if v: agg[k]=sum(v)/len(v)
        agg["recall_sd"]=(sum((r["recall_arc"]-agg["recall_arc"])**2 for r in runs)/len(runs))**0.5
        results["arms"][arm]=agg
        def gg(k,fmt="%.3f",w=7): return (fmt%agg[k]).rjust(w) if k in agg else "   -  ".rjust(w)
        print(f"{arm:20s} {agg['recall_arc']:6.3f}±{agg['recall_sd']:.3f} {agg['bins']:8.1f} "
              f"{gg('T_onman',w=8)} {gg('T_shift_deg','%.1f',7)} {gg('T_shift_std','%.1f',8)} {gg('M_angle','%.1f',7)} {gg('M_off','%.2f',6)}")
    print(f"\nideal recall_arc={45/360:.3f}. WIN = FROZEN_LATENT_* fills arc with LOW shift_std (coherent) vs FROZEN_RESIDUAL incoherent.")
    os.makedirs(os.path.dirname(args.out_json) or ".",exist_ok=True)
    json.dump(results,open(args.out_json,"w"),indent=2); print(f"wrote {args.out_json}")

if __name__=="__main__": main()
