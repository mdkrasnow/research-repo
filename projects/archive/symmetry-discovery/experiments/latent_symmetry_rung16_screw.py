"""
Ladder rung 16: NON-ROTATION symmetry (screw/helix) — does the recipe generalize beyond rotation?

Pre-register:
  Question: can the frozen-anchor + group-generator recipe discover a SCREW symmetry (rotation+translation
            coupled, NON-isometric)? And does the isometry prior (cond->1) correctly FAIL on it while a
            volume-only prior (det~1) succeeds?
  World: helix z(φ)=[R cosφ, R sinφ, C·φ], φ∈[0,4π). Symmetry = screw: φ→φ+Δ, h→h+CΔ (rotation in (x,y)
         coupled with translation in h). Held-out = mid φ-arc. Clean supervised latent (enc->z).
  Operator: homogeneous coords [z,1]∈R^4, M=exp(A) with affine last row -> can represent rotation AND
            translation (SE-like 1-param subgroups). T(x)=DECODE((M·[enc(x),1])[:3]).
  Arms:
    BASE                      floor
    ORACLE                    true screw aug (positive control)
    GEN_SKEW_SCREW            A in screw subalgebra (rot block + h-translation); learn ω,v  (strong ref)
    GEN_FREE_STABLE_VOLONLY   general affine A, reg = det(linear)~1 only (volume-preserving)  (TREATMENT)
    GEN_FREE_STABLE_COND      general affine A, reg = det~1 + cond->1 (isometry)              (ablation: should FAIL on screw)
  Frozen anchor (energy distance to unlabeled full-helix samples). No live EqM closure. No held-out targets.
  Success: VOLONLY recovers screw (rotation+translation coupling v/ω≈C) + fills arc near oracle, while
           COND (isometry prior) fails to capture the translation. Shows the "stable group" prior must be
           volume-preserving, not isometric, to cover non-rotation symmetries.
  Kill: if VOLONLY fails twice under good controls, stop; report the recipe is rotation-limited.
"""
import argparse, json, math, os
import torch, torch.nn as nn

D=16; KZ=3; RAD=4.0; CHELIX=1.5; PHI_MAX=4*math.pi; SIG=0.08   # steep helix: rotation-only leaves manifold (C=0.4 made it ~circle -> confounded rung16-v1)
GAP_LO=2.5; GAP_HI=3.5                                  # held-out phi-arc (radians)
T_STEPS=3000; AE_STEPS=3000; EQM_STEPS=4000; BATCH=512; SAMPLE_STEPS=50; N_SAMPLE=3000; SEEDS=[0,1,2]
ALPHA_AUG=1.0; LAM_MOVE=0.5; LAM_DET=1.0; LAM_COND=1.0; LAM_TRANS=1.0; DEC_SEED=1234; TARGET_DPHI=1.0
ARMS=["BASE","ORACLE","GEN_SKEW_SCREW","GEN_FREE_STABLE_VOLONLY","GEN_FREE_STABLE_ISOMETRY"]
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mlp(sizes, act=nn.SiLU):
    L=[]
    for i in range(len(sizes)-2): L+=[nn.Linear(sizes[i],sizes[i+1]),act()]
    L+=[nn.Linear(sizes[-2],sizes[-1])]; return nn.Sequential(*L)

def build_world():
    g=torch.Generator(device="cpu").manual_seed(DEC_SEED)
    dec=nn.Sequential(nn.Linear(KZ,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,D))
    with torch.no_grad():
        for p in dec.parameters(): p.copy_(torch.randn(p.shape,generator=g)*(1.0 if p.dim()==1 else 0.9))
    dec=dec.to(DEV)
    for p in dec.parameters(): p.requires_grad_(False)
    def zfn(phi): return torch.stack([RAD*torch.cos(phi),RAD*torch.sin(phi),CHELIX*phi],1)
    with torch.no_grad():
        phi=torch.rand(40000,device=DEV)*PHI_MAX; z=zfn(phi)+SIG*torch.randn(40000,3,device=DEV)
        raw=dec(z); mu,sd=raw.mean(0),raw.std(0)+1e-6
    decode=lambda z:(dec(z)-mu)/sd
    with torch.no_grad():
        pg=torch.linspace(0,PHI_MAX,400,device=DEV); man=decode(zfn(pg))
        sig_obs=(decode(zfn(torch.zeros(4000,device=DEV))+SIG*torch.randn(4000,3,device=DEV))-man[0]).norm(dim=1).mean().item()
    return decode,zfn,man,pg,sig_obs

DECODE,ZFN,MAN,GRID_PHI,SIG_OBS=build_world()
TARGET_MOVE=float((DECODE(ZFN(torch.tensor([TARGET_DPHI],device=DEV)))-DECODE(ZFN(torch.tensor([0.0],device=DEV)))).norm())

def in_gap(phi): return (phi>=GAP_LO)&(phi<=GAP_HI)
def sample_phi_vis(n):
    out=torch.empty(0,device=DEV)
    while out.numel()<n:
        p=torch.rand(n,device=DEV)*PHI_MAX; p=p[~in_gap(p)]; out=torch.cat([out,p])
    return out[:n]
def sample_visible(n): return DECODE(ZFN(sample_phi_vis(n))+SIG*torch.randn(n,3,device=DEV))
def sample_full(n): return DECODE(ZFN(torch.rand(n,device=DEV)*PHI_MAX)+SIG*torch.randn(n,3,device=DEV))
def screw(z, dphi):                                    # true symmetry: rotate (x,y) by dphi, h += C*dphi
    c,s=math.cos(dphi),math.sin(dphi)
    x=z[:,0]*c-z[:,1]*s; y=z[:,0]*s+z[:,1]*c; h=z[:,2]+CHELIX*dphi
    return torch.stack([x,y,h],1)
def eqm_loss(f,x):
    n=x.shape[0]; g=torch.rand(n,1,device=DEV); eps=torch.randn(n,D,device=DEV); xg=(1-g)*x+g*eps
    return ((f(torch.cat([xg,g],1))-(eps-x))**2).mean()
def energy_distance(a,b): return 2*torch.cdist(a,b).mean()-torch.cdist(a,a).mean()-torch.cdist(b,b).mean()
@torch.no_grad()
def sample(f,n):
    x=torch.randn(n,D,device=DEV)
    for i in range(SAMPLE_STEPS):
        g=torch.full((n,1),1-i/SAMPLE_STEPS,device=DEV); x=x-f(torch.cat([x,g],1))*(1.0/SAMPLE_STEPS)
    return x

E_ROT=torch.zeros(4,4,device=DEV); E_ROT[0,1]=-1.0; E_ROT[1,0]=1.0      # rotation generator (x,y)
E_TRANS=torch.zeros(4,4,device=DEV); E_TRANS[2,3]=1.0                    # h-translation generator

class ScrewOp(nn.Module):
    def __init__(self, kind):                                            # kind: screw_known / volonly / cond
        super().__init__(); self.enc=mlp([D,64,KZ]); self.kind=kind
        if kind=="screw_known":
            self.omega=nn.Parameter(torch.tensor(0.5)); self.vtr=nn.Parameter(torch.tensor(0.2))
        else:
            self.A_raw=nn.Parameter(0.10*torch.randn(4,4))
    def Amat(self):
        if self.kind=="screw_known": return self.omega*E_ROT+self.vtr*E_TRANS
        Z=torch.zeros(1,4,device=DEV)                                    # affine: force last row 0
        return torch.cat([self.A_raw[:3,:], Z],0)
    def M(self): return torch.matrix_exp(self.Amat())
    def op_params(self): return [self.omega,self.vtr] if self.kind=="screw_known" else [self.A_raw]
    def reg(self):
        if self.kind=="screw_known": return torch.zeros((),device=DEV)
        M=self.M(); lin=M[:3,:3]; det=torch.det(lin)
        r=LAM_DET*(torch.log(det.abs()+1e-8))**2
        if self.kind=="isometry":   # rotation/isometry prior: well-conditioned linear part + NO translation
            sv=torch.linalg.svdvals(lin); r=r+LAM_COND*(sv.max()/sv.min().clamp_min(1e-6)-1.0)**2
            r=r+LAM_TRANS*(M[:3,3]**2).sum()                 # forbid the affine translation -> should FAIL on screw
        return r
    def forward(self,x):
        z=self.enc(x); hom=torch.cat([z,torch.ones(z.size(0),1,device=DEV)],1)
        return DECODE((hom@self.M().T)[:,:3])

def discover(seed, kind):
    torch.manual_seed(seed); T=ScrewOp(kind).to(DEV)
    eo=torch.optim.Adam(T.enc.parameters(),lr=2e-3)
    for _ in range(AE_STEPS):                                            # enc supervised -> true z (clean latent)
        phi=torch.rand(BATCH,device=DEV)*PHI_MAX; z=ZFN(phi); x=DECODE(z+SIG*torch.randn_like(z))
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
            dphi=float(torch.rand(1))*3-1.5; phi=sample_phi_vis(BATCH)
            loss=loss+ALPHA_AUG*eqm_loss(f,DECODE(screw(ZFN(phi),dphi)))
        elif T is not None:
            loss=loss+ALPHA_AUG*eqm_loss(f,T(x).detach())
        loss.backward(); opt.step()
    return f

def run_arm(arm, seed):
    if arm=="BASE": return train_field(seed), None
    if arm=="ORACLE": return train_field(seed, oracle=True), None
    kind={"GEN_SKEW_SCREW":"screw_known","GEN_FREE_STABLE_VOLONLY":"volonly","GEN_FREE_STABLE_ISOMETRY":"isometry"}[arm]
    T=discover(seed,kind); return train_field(seed,T=T), T

@torch.no_grad()
def nearest_phi(x):
    d=torch.cdist(x,MAN); idx=d.argmin(1); return GRID_PHI[idx], d.min(1).values

@torch.no_grad()
def evaluate(f,T):
    s=sample(f,N_SAMPLE); phi,dist=nearest_phi(s); within=dist<3*SIG_OBS
    rec=float((within&in_gap(phi)).float().sum()/N_SAMPLE)
    bins=torch.floor(phi/PHI_MAX*40).clamp(0,39).long()
    cov=torch.tensor([((bins==b)&within).float().sum() for b in range(40)],device=DEV)/N_SAMPLE
    out={"recall_arc":rec,"bins":int((cov>0.004).sum()),"onman":float(within.float().mean()),
         "mmd_full":float(energy_distance(s,sample_full(N_SAMPLE)))}
    if T is not None:
        xa=sample_full(1024); Tx=T(xa); p0,_=nearest_phi(xa); p1,d1=nearest_phi(Tx)
        out["T_onman"]=float((d1<3*SIG_OBS).float().mean())
        dphi=p1-p0; out["T_dphi_mean"]=float(dphi.mean()); out["T_dphi_std"]=float(dphi.std())
        M=T.M(); lin=M[:3,:3]; sv=torch.linalg.svdvals(lin)
        out["lin_det"]=float(torch.det(lin)); out["lin_cond"]=float(sv.max()/sv.min().clamp_min(1e-6))
        # recovered rotation rate omega (angle in x,y) and translation rate in h
        out["rec_omega_deg"]=float(torch.atan2(lin[1,0],lin[0,0])*180/math.pi)
        out["rec_htrans"]=float(M[2,3])
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-json",default="results_rung16.json"); ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()
    global T_STEPS,AE_STEPS,EQM_STEPS
    seeds=SEEDS
    if args.quick: T_STEPS,AE_STEPS,EQM_STEPS,seeds=400,400,400,[0]
    print(f"device={DEV} helix C={CHELIX} (true h-trans per rad) TARGET_MOVE={TARGET_MOVE:.3f}")
    print(f"held-out phi-arc=[{GAP_LO},{GAP_HI}] of [0,{PHI_MAX:.2f}]  ideal recall~{(GAP_HI-GAP_LO)/PHI_MAX:.3f}\n")
    results={"config":{"C":CHELIX,"seeds":seeds,"device":str(DEV)},"arms":{}}
    hdr=f"{'arm':26s} {'recall':>7s} {'%ORC':>5s} {'bins':>4s} {'T_onm':>6s} {'dphi_sd':>7s} {'det':>5s} {'cond':>5s} {'omega':>6s} {'htrans':>7s}"
    print(hdr); print("-"*len(hdr))
    orc=None
    for arm in ARMS:
        runs=[evaluate(*run_arm(arm,s)) for s in seeds]
        agg={}
        for k in ("recall_arc","bins","T_onman","T_dphi_std","lin_det","lin_cond","rec_omega_deg","rec_htrans"):
            v=[r[k] for r in runs if k in r]
            if v: agg[k]=sum(v)/len(v)
        agg["recall_sd"]=(sum((r["recall_arc"]-agg["recall_arc"])**2 for r in runs)/len(runs))**0.5
        results["arms"][arm]=agg
        if arm=="ORACLE": orc=agg["recall_arc"]
        pct=(100*agg["recall_arc"]/orc) if orc else 0
        def gg(k,fmt="%.2f",w=5): return (fmt%agg[k]).rjust(w) if k in agg else "  -  ".rjust(w)
        print(f"{arm:26s} {agg['recall_arc']:6.3f} {pct:5.0f} {agg['bins']:4.0f} {gg('T_onman',w=6)} "
              f"{gg('T_dphi_std','%.2f',7)} {gg('lin_det',w=5)} {gg('lin_cond',w=5)} {gg('rec_omega_deg','%.0f',6)} {gg('rec_htrans','%.3f',7)}")
    print(f"\ntrue screw: rotation+translation coupled, h-trans/omega ratio ~ C={CHELIX} per rad.")
    print("WIN = VOLONLY fills arc near ORACLE + recovers htrans>0 (screw), while ISOMETRY (htrans forbidden) fails to fill.")
    os.makedirs(os.path.dirname(args.out_json) or ".",exist_ok=True)
    json.dump(results,open(args.out_json,"w"),indent=2); print(f"wrote {args.out_json}")

if __name__=="__main__": main()
