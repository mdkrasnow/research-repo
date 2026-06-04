"""
Ladder rung 9: FROZEN-ANCHOR latent symmetry discovery.

Terminal diagnosis from rungs 5-8: discovering T via EqM closure through the LIVE field fails because
the field co-adapts to whatever T emits -- closure goes down without T being manifold-preserving.

Fix tested here: anchor T to a FROZEN, non-co-adapting manifold reference, learn T BEFORE EqM, freeze
T, then use T as augmentation for EqM. Minimal anchor = energy-distance to UNLABELED samples from the
FULL observed manifold (incl. the held-out arc as unlabeled support; NO EqM targets there, NO latent
labels). Fair question: given unlabeled full-manifold data, can we learn a reusable manifold-preserving
operator that transfers EqM supervision into regions EqM was never trained on?

Three data roles:
  anchor   : unlabeled samples from FULL observed manifold (defines the frozen reference)
  eqm-train: observed samples from VISIBLE subset only (held-out arc removed)
  eval     : full manifold incl. held-out arc (metrics only)

Arms:
  BASE                  EqM on visible only                                  (floor)
  ORACLE                EqM + true-group (random SO(2)) augmentation         (positive control)
  FIELD_CLOSURE_DISC    learn T via live-field EqM closure (old failed style) (reproduce failure)
  FROZEN_ANCHOR_DISC    learn T vs FROZEN full-manifold anchor, freeze, aug   (THE TREATMENT)
  VISIBLE_ANCHOR_DISC   same but anchor = VISIBLE-only samples                (leakage/negative control)

Central question: does the frozen full-manifold anchor let T learn a reusable manifold-preserving
operator that then fills the held-out arc? Clean win = FROZEN beats FIELD_CLOSURE and VISIBLE.
"""
import argparse, json, math, os
import torch, torch.nn as nn

D=16; RAD=4.0; SIG=0.08
GAP_LO=157.5*math.pi/180; GAP_HI=202.5*math.pi/180     # held-out arc (45deg)
T_STEPS=3000; EQM_STEPS=4000; BATCH=512; SAMPLE_STEPS=50; N_SAMPLE=3000; SEEDS=[0,1,2]
ALPHA_AUG=1.0; LAM_MOVE=0.5; W_WD=1e-4; DEC_SEED=1234; T_RES_SCALE=0.3
TARGET_DEG=60                                          # move ~60deg (> 45deg arc so aug covers it)
ARMS=["BASE","ORACLE","FIELD_CLOSURE_DISC","FROZEN_ANCHOR_DISC","VISIBLE_ANCHOR_DISC"]
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
TARGET_MOVE=float((MAN[TARGET_DEG]-MAN[0]).norm())     # observed chord for ~TARGET_DEG rotation

def in_gap(phi): return (phi>=GAP_LO)&(phi<=GAP_HI)
def z_of(phi): return torch.stack([RAD*torch.cos(phi),RAD*torch.sin(phi)],1)
def sample_visible(n):
    out=torch.empty(0,device=DEV)
    while out.numel()<n:
        p=torch.rand(n,device=DEV)*2*math.pi; p=p[~in_gap(p)]; out=torch.cat([out,p])
    phi=out[:n]; return DECODE(z_of(phi)+SIG*torch.randn(n,2,device=DEV))
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

# ---------------- T discovery ----------------
def train_T_anchor(seed, anchor_sampler):
    """Learn T against a FROZEN distributional anchor (energy distance). No field involved."""
    torch.manual_seed(seed); T=ResidualT().to(DEV)
    opt=torch.optim.Adam(T.parameters(),lr=2e-3,weight_decay=W_WD)
    for _ in range(T_STEPS):
        xa=anchor_sampler(BATCH); xb=anchor_sampler(BATCH)
        tx=T(xa); opt.zero_grad()
        move=(tx-xa).norm(dim=1).mean()
        loss=energy_distance(tx,xb)+LAM_MOVE*(move-TARGET_MOVE)**2
        loss.backward(); opt.step()
    for p in T.parameters(): p.requires_grad_(False)
    return T

def train_field(seed, T=None, oracle=False):
    torch.manual_seed(seed); f=mlp([D+1,256,256,D]).to(DEV)
    opt=torch.optim.Adam(f.parameters(),lr=2e-3)
    for _ in range(EQM_STEPS):
        x=sample_visible(BATCH); opt.zero_grad(); loss=eqm_loss(f,x)
        if oracle:
            th=float(torch.rand(1))*2*math.pi
            phi=torch.rand(BATCH,device=DEV)*2*math.pi; phi=phi[~in_gap(phi)]
            z=z_of(phi)@rot_mat(th).T; loss=loss+ALPHA_AUG*eqm_loss(f,DECODE(z))
        elif T is not None:
            loss=loss+ALPHA_AUG*eqm_loss(f,T(x).detach())
        loss.backward(); opt.step()
    return f

def train_field_closure(seed):
    """OLD failed style: co-train T with the LIVE field via EqM closure. Reproduce the failure."""
    torch.manual_seed(seed); f=mlp([D+1,256,256,D]).to(DEV); T=ResidualT().to(DEV)
    opt=torch.optim.Adam(list(f.parameters())+list(T.parameters()),lr=2e-3,weight_decay=W_WD)
    for _ in range(EQM_STEPS):
        x=sample_visible(BATCH); opt.zero_grad()
        tx=T(x); move=(tx-x).norm(dim=1).mean()
        loss=eqm_loss(f,x)+ALPHA_AUG*eqm_loss(f,tx)+LAM_MOVE*(move-TARGET_MOVE)**2
        loss.backward(); opt.step()
    for p in T.parameters(): p.requires_grad_(False)
    return f,T

def run_arm(arm, seed):
    if arm=="BASE": return train_field(seed), None
    if arm=="ORACLE": return train_field(seed, oracle=True), None
    if arm=="FIELD_CLOSURE_DISC": f,T=train_field_closure(seed); return f,T
    if arm=="FROZEN_ANCHOR_DISC":
        T=train_T_anchor(seed, sample_full); return train_field(seed,T=T), T
    if arm=="VISIBLE_ANCHOR_DISC":
        T=train_T_anchor(seed, sample_visible); return train_field(seed,T=T), T

# ---------------- metrics ----------------
@torch.no_grad()
def evaluate(f,T):
    s=sample(f,N_SAMPLE); d=torch.cdist(s,MAN); near=d.argmin(1); within=d.min(1).values<3*SIG_OBS
    phi=PHI_GRID[near]
    rec=float((within&in_gap(phi)).float().sum()/N_SAMPLE)
    bins=torch.floor(phi/(2*math.pi)*36).clamp(0,35).long()
    cov=torch.tensor([((bins==b)&within).float().sum() for b in range(36)],device=DEV)/N_SAMPLE
    ref=sample_full(N_SAMPLE)
    out={"recall_arc":rec,"bins":int((cov>0.005).sum()),"onman_gen":float(within.float().mean()),
         "mmd_full":float(energy_distance(s,ref))}
    if T is not None:
        xa=sample_full(1024); Tx=T(xa)
        out["T_onman"]=float((torch.cdist(Tx,MAN).min(1).values<3*SIG_OBS).float().mean())
        out["T_move"]=float((Tx-xa).norm(dim=1).mean()/TARGET_MOVE)
        # shift consistency: angular shift induced by T, std small => reusable consistent operator
        p0=PHI_GRID[torch.cdist(xa,MAN).argmin(1)]; p1=PHI_GRID[torch.cdist(Tx,MAN).argmin(1)]
        dphi=((p1-p0+math.pi)%(2*math.pi))-math.pi
        out["T_shift_deg"]=float(dphi.mean()*180/math.pi); out["T_shift_std"]=float(dphi.std()*180/math.pi)
        # orbit: repeated application, distinct bins visited
        xk=sample_full(1)[:1]; vis=set()
        for _ in range(12):
            xk=T(xk); vis.add(int(torch.floor(PHI_GRID[torch.cdist(xk,MAN).argmin(1)][0]/(2*math.pi)*36)))
        out["T_orbit_bins"]=len(vis)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-json",default="results_rung9.json")
    ap.add_argument("--quick",action="store_true"); args=ap.parse_args()
    global T_STEPS,EQM_STEPS
    seeds=SEEDS
    if args.quick: T_STEPS,EQM_STEPS,seeds=400,400,[0]
    print(f"device={DEV} arc=[157.5,202.5]deg TARGET_MOVE({TARGET_DEG}deg)={TARGET_MOVE:.3f} sig_obs={SIG_OBS:.3f}")
    print(f"ideal recall_arc={45/360:.3f}\n")
    results={"config":{"T_steps":T_STEPS,"eqm_steps":EQM_STEPS,"seeds":seeds,"device":str(DEV)},"arms":{}}
    hdr=f"{'arm':20s} {'recall_arc':>10s} {'bins/36':>8s} {'mmd_full':>9s} {'T_onman':>8s} {'T_move':>7s} {'shift°':>7s} {'shiftSD':>8s} {'orbit':>6s}"
    print(hdr); print("-"*len(hdr))
    for arm in ARMS:
        runs=[evaluate(*run_arm(arm,s)) for s in seeds]
        agg={}
        for k in ("recall_arc","bins","mmd_full","onman_gen","T_onman","T_move","T_shift_deg","T_shift_std","T_orbit_bins"):
            v=[r[k] for r in runs if k in r]
            if v: agg[k]=sum(v)/len(v)
        agg["recall_sd"]=(sum((r["recall_arc"]-agg["recall_arc"])**2 for r in runs)/len(runs))**0.5
        results["arms"][arm]=agg
        def gg(k,fmt="%.3f",w=7): return (fmt%agg[k]).rjust(w) if k in agg else "   -  ".rjust(w)
        print(f"{arm:20s} {agg['recall_arc']:6.3f}±{agg['recall_sd']:.3f} {agg['bins']:8.1f} {gg('mmd_full','%.3f',9)} "
              f"{gg('T_onman',w=8)} {gg('T_move')} {gg('T_shift_deg','%.1f',7)} {gg('T_shift_std','%.1f',8)} {gg('T_orbit_bins','%.0f',6)}")
    print(f"\nideal recall_arc={45/360:.3f}. WIN = FROZEN_ANCHOR_DISC fills arc (recall up, T_onman high,")
    print("consistent shift, orbit visits many bins) while FIELD_CLOSURE_DISC and VISIBLE_ANCHOR_DISC do not.")
    os.makedirs(os.path.dirname(args.out_json) or ".",exist_ok=True)
    json.dump(results,open(args.out_json,"w"),indent=2); print(f"\nwrote {args.out_json}")

if __name__=="__main__": main()
