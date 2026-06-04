"""
Ladder rung 14: HIGHER-DIM prior-budget test (K=3), where stable != rotation.

Rung 13 caveat: in 2D, det~1 + well-conditioned => orthogonal => rotation, so "stable" and "rotation"
nearly coincide. Rung 14 breaks that: latent is a CYLINDER in R^3 — z = [r cos phi, r sin phi, h] — with
hidden symmetry  phi -> phi + Delta,  h UNCHANGED. Now a general stable 3x3 operator has many
non-rotation options (it could mix/translate h); the correct symmetry is rotation CONFINED to the (0,1)
plane with h preserved. So the stability prior is genuinely broader than the answer.

Central question: does GEN_FREE_STABLE (general 3x3, M=exp(A), stability-only reg: det~1 + cond->1,
NO skew, NO isometry term) discover rotation-in-the-active-plane-with-h-preserved, reaching oracle,
WITHOUT being told the plane or the symmetry class?

Arms:
  BASE                   EqM visible-only                                           (floor)
  ORACLE                 EqM + true symmetry aug (random phi-rotation, h fixed)     (positive control)
  GEN_SKEW_KNOWN_PLANE   skew restricted to true (0,1) plane; learn angle           (strong upper ref)
  GEN_SKEW_FREE_PLANE    skew 3x3 (learn the plane/axis), M=exp(skew(A))            (rotation-class prior)
  GEN_FREE_STABLE        general 3x3, M=exp(A) + det~1 + cond->1 (NO skew)          (the key treatment)

Frozen anchor (energy distance to unlabeled FULL-manifold samples). Clean latent (enc supervised->z, 3D)
to isolate the operator-prior question. No live-field closure, no held-out targets, no mining.
Held-out = phi-arc (all h). Key new diagnostic: passive_leak (does M disturb h?) + active-plane alignment.
"""
import argparse, json, math, os
import torch, torch.nn as nn

D=16; K_LAT=3; RAD=4.0; HRANGE=1.5; SIG=0.08
GAP_LO=157.5*math.pi/180; GAP_HI=202.5*math.pi/180
T_STEPS=3000; AE_STEPS=3000; EQM_STEPS=4000; BATCH=512; SAMPLE_STEPS=50; N_SAMPLE=3000; SEEDS=[0,1,2]
ALPHA_AUG=1.0; LAM_MOVE=0.5; LAM_DET=1.0; LAM_COND=1.0; DEC_SEED=1234; TARGET_DEG=60
ARMS=["BASE","ORACLE","GEN_SKEW_KNOWN_PLANE","GEN_SKEW_FREE_PLANE","GEN_FREE_STABLE"]
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
        man=decode(torch.stack([RAD*torch.cos(PP),RAD*torch.sin(PP),HH],1))     # [504,D]
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

E01=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]],device=DEV)   # generator of rotation in the true (0,1) plane

class LatentOp(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.enc=mlp([D,64,K_LAT]); self.kind=kind
        if kind=="skew_known": self.theta=nn.Parameter(torch.tensor(1.0))
        elif kind=="skew_free": self.A=nn.Parameter(0.3*torch.randn(K_LAT,K_LAT))
        elif kind=="stable":    self.A=nn.Parameter(0.10*torch.randn(K_LAT,K_LAT))
    def M(self):
        if self.kind=="skew_known": return torch.matrix_exp(self.theta*E01)
        if self.kind=="skew_free":  return torch.matrix_exp(0.5*(self.A-self.A.T))
        return torch.matrix_exp(self.A)                                  # stable
    def op_params(self):
        return [self.theta] if self.kind=="skew_known" else [self.A]
    def reg(self):
        if self.kind!="stable": return torch.zeros((),device=DEV)
        M=self.M(); det=torch.det(M); sv=torch.linalg.svdvals(M)
        cond=(sv.max()/sv.min().clamp_min(1e-6)-1.0)**2
        return LAM_DET*(torch.log(det.abs()+1e-8))**2 + LAM_COND*cond     # det~1 + well-conditioned ONLY
    def forward(self,x): return DECODE(self.enc(x)@self.M().T)

def anchor_train_op(seed, kind):
    torch.manual_seed(seed); T=LatentOp(kind).to(DEV)
    eo=torch.optim.Adam(T.enc.parameters(),lr=2e-3)
    for _ in range(AE_STEPS):
        phi=torch.rand(BATCH,device=DEV)*2*math.pi; z=z_of(phi,sample_h(BATCH)); x=DECODE(z+SIG*torch.randn_like(z))
        eo.zero_grad(); ((T.enc(x)-z)**2).mean().backward(); eo.step()
    for p in T.enc.parameters(): p.requires_grad_(False)
    opt=torch.optim.Adam(T.op_params(),lr=2e-3)
    for _ in range(T_STEPS):
        xa=sample_full(BATCH); xb=sample_full(BATCH); tx=T(xa); opt.zero_grad()
        move=(tx-xa).norm(dim=1).mean()
        (energy_distance(tx,xb)+LAM_MOVE*(move-TARGET_MOVE)**2+T.reg()).backward(); opt.step()
    for p in T.op_params(): p.requires_grad_(False)
    return T

def train_field(seed, T=None, oracle=False):
    torch.manual_seed(seed); f=mlp([D+1,256,256,D]).to(DEV); opt=torch.optim.Adam(f.parameters(),lr=2e-3)
    for _ in range(EQM_STEPS):
        x=sample_visible(BATCH); opt.zero_grad(); loss=eqm_loss(f,x)
        if oracle:
            th=float(torch.rand(1))*2*math.pi; phi=torch.rand(BATCH,device=DEV)*2*math.pi; phi=phi[~in_gap(phi)]
            loss=loss+ALPHA_AUG*eqm_loss(f,DECODE(z_of(phi+th, sample_h(phi.shape[0]))))   # rotate phi, h free
        elif T is not None:
            loss=loss+ALPHA_AUG*eqm_loss(f,T(x).detach())
        loss.backward(); opt.step()
    return f

def run_arm(arm, seed):
    if arm=="BASE": return train_field(seed), None
    if arm=="ORACLE": return train_field(seed, oracle=True), None
    kind={"GEN_SKEW_KNOWN_PLANE":"skew_known","GEN_SKEW_FREE_PLANE":"skew_free","GEN_FREE_STABLE":"stable"}[arm]
    T=anchor_train_op(seed,kind); return train_field(seed,T=T), T

@torch.no_grad()
def nearest(x):
    idx=torch.cdist(x,MAN).argmin(1); return GRID_PHI[idx], GRID_H[idx], torch.cdist(x,MAN).min(1).values

@torch.no_grad()
def evaluate(f,T):
    s=sample(f,N_SAMPLE); phi,h,dist=nearest(s); within=dist<3*SIG_OBS
    rec=float((within&in_gap(phi)).float().sum()/N_SAMPLE)
    bins=torch.floor(phi/(2*math.pi)*36).clamp(0,35).long()
    cov=torch.tensor([((bins==b)&within).float().sum() for b in range(36)],device=DEV)/N_SAMPLE
    out={"recall_arc":rec,"bins":int((cov>0.005).sum()),"onman_gen":float(within.float().mean()),
         "mmd_full":float(energy_distance(s,sample_full(N_SAMPLE)))}
    if T is not None:
        xa=sample_full(1024); Tx=T(xa)
        p0,h0,_=nearest(xa); p1,h1,d1=nearest(Tx)
        out["T_onman"]=float((d1<3*SIG_OBS).float().mean())
        dphi=((p1-p0+math.pi)%(2*math.pi))-math.pi
        out["T_shift_deg"]=float(dphi.mean()*180/math.pi); out["T_shift_std"]=float(dphi.std()*180/math.pi)
        out["passive_dh"]=float((h1-h0).abs().mean())                 # should be ~0 if h preserved
        M=T.M(); I=torch.eye(K_LAT,device=DEV)
        out["M_det"]=float(torch.det(M))
        sv=torch.linalg.svdvals(M); out["M_cond"]=float(sv.max()/sv.min().clamp_min(1e-6))
        out["M_angle"]=float(torch.atan2(M[1,0],M[0,0])*180/math.pi)   # angle in true active plane
        # passive-leak: how much M couples the h-axis (index 2) to the active plane
        out["passive_leak"]=float(M[2,:2].norm()+M[:2,2].norm()+abs(M[2,2]-1))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-json",default="results_rung14.json"); ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()
    global T_STEPS,AE_STEPS,EQM_STEPS
    seeds=SEEDS
    if args.quick: T_STEPS,AE_STEPS,EQM_STEPS,seeds=400,400,400,[0]
    print(f"device={DEV} K_LAT={K_LAT} (cylinder: rot in plane(0,1), h passive) TARGET_MOVE={TARGET_MOVE:.3f} ideal recall_arc={45/360:.3f}\n")
    results={"config":{"K_LAT":K_LAT,"seeds":seeds,"device":str(DEV)},"arms":{}}
    hdr=f"{'arm':22s} {'recall':>8s} {'%ORC':>5s} {'bins':>4s} {'T_onm':>6s} {'shiftSD':>7s} {'pass_dh':>7s} {'leak':>5s} {'det':>5s} {'cond':>5s} {'ang':>6s}"
    print(hdr); print("-"*len(hdr))
    orc=None
    for arm in ARMS:
        runs=[evaluate(*run_arm(arm,s)) for s in seeds]
        agg={}
        for k in ("recall_arc","bins","onman_gen","T_onman","T_shift_std","passive_dh","passive_leak","M_det","M_cond","M_angle"):
            v=[r[k] for r in runs if k in r]
            if v: agg[k]=sum(v)/len(v)
        agg["recall_sd"]=(sum((r["recall_arc"]-agg["recall_arc"])**2 for r in runs)/len(runs))**0.5
        results["arms"][arm]=agg
        if arm=="ORACLE": orc=agg["recall_arc"]
        pct=(100*agg["recall_arc"]/orc) if orc else 0
        def gg(k,fmt="%.2f",w=5): return (fmt%agg[k]).rjust(w) if k in agg else "  -  ".rjust(w)
        print(f"{arm:22s} {agg['recall_arc']:6.3f}  {pct:4.0f} {agg['bins']:4.0f} {gg('T_onman',w=6)} "
              f"{gg('T_shift_std','%.1f',7)} {gg('passive_dh','%.3f',7)} {gg('passive_leak','%.2f',5)} "
              f"{gg('M_det',w=5)} {gg('M_cond',w=5)} {gg('M_angle','%.0f',6)}")
    print(f"\nideal recall_arc={45/360:.3f}. KEY: does GEN_FREE_STABLE reach ORACLE with low passive_dh/leak")
    print("(rotation confined to active plane, h preserved) WITHOUT skew or knowing the plane?")
    os.makedirs(os.path.dirname(args.out_json) or ".",exist_ok=True)
    json.dump(results,open(args.out_json,"w"),indent=2); print(f"wrote {args.out_json}")

if __name__=="__main__": main()
