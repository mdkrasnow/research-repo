"""
Ladder rung 7: CONTINUOUS manifold so continuity forces the symmetry.

Rung 6 + force-balance diagnostic finding: with DISCRETE modes, gradient-based symmetry
discovery is ill-posed -- intermediate rotations (15-44deg) land samples BETWEEN modes
(off-manifold) = energy barrier, so M cannot walk from identity to the true 45deg. Even with a
perfect clean latent (ORACLE_LATENT) and a 20x anti-identity push, M stalled at ~-15deg.

Fix: a CONTINUOUS ring (dense angle phi, held-out ARC) decoded by the same nonlinear g. Now
rotating phi stays ON the manifold the whole way (circle is rotation-closed) -> smooth on-manifold
path identity->symmetry, no barrier -> M can continuously rotate to discover it; continuity across
the gap forces the symmetry to extend into the held-out arc.

Manifold: z = RAD*(cos phi, sin phi) + small noise. Held-out arc phi in [157.5, 202.5] deg (45 wide,
matches cyclic order 8 so one 45deg rotation fills it exactly). Train = the other 315deg.
Symmetry group = SO(2). recall@heldout = fraction of generated samples whose nearest manifold angle
lands in the held-out arc (ideal = 45/360 = 0.125).

Arms (same logic as rung 5/6):
  BASE          floor
  ORACLE        + aug with RANDOM rotations (full SO(2) group) -> positive control, fills arc
  DISC_LINEAR   linear obs-space op -> negative control
  ORACLE_LATENT enc supervised->z (clean latent), frozen, learn M  -> does discovery work given
                a clean CONTINUOUS latent? (the gate: must recover a rotation + fill the arc)
  DISC_JOINT    enc/dec/M/field learned (unsupervised) -> the real test
Recipe: closure-aug + hinge anti-identity + cyclic M^P~I (P=8). Low prior (k=2, order~8).
"""
import argparse, json, math, os
import torch, torch.nn as nn

D=16; K_LAT=2; RAD=4.0; SIG=0.08
GAP_LO=157.5*math.pi/180; GAP_HI=202.5*math.pi/180          # held-out arc (45deg)
STEPS=4000; AE_STEPS=3000; BATCH=512; SAMPLE_STEPS=50; N_SAMPLE=3000; SEEDS=[0,1,2]
W_AUG=1.0; W_MOVE=1.0; W_CYC=1.0; W_RECON=1.0; P_CYCLE=8; W_WD=1e-4; DEC_SEED=1234
ARMS=["BASE","ORACLE","DISC_LINEAR","ORACLE_LATENT","DISC_JOINT"]
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
        man=decode(torch.stack([RAD*torch.cos(pg),RAD*torch.sin(pg)],1))      # [360,D] manifold grid
        # observed noise scale
        z0=torch.stack([RAD*torch.cos(torch.zeros(4000,device=DEV)),RAD*torch.sin(torch.zeros(4000,device=DEV))],1)+SIG*torch.randn(4000,2,device=DEV)
        sig_obs=(decode(z0)-man[0]).norm(dim=1).mean().item()
    return decode,pg,man,sig_obs

DECODE,PHI_GRID,MAN,SIG_OBS=build_world()

def in_gap(phi): return (phi>=GAP_LO)&(phi<=GAP_HI)

def sample_phi_train(n):
    out=torch.empty(0,device=DEV)
    while out.numel()<n:
        p=torch.rand(n,device=DEV)*2*math.pi
        p=p[~in_gap(p)]
        out=torch.cat([out,p])
    return out[:n]

def z_of(phi): return torch.stack([RAD*torch.cos(phi),RAD*torch.sin(phi)],1)
def sample_train(n): return DECODE(z_of(sample_phi_train(n))+SIG*torch.randn(n,2,device=DEV))

def rot_mat(theta):
    c,s=math.cos(theta),math.sin(theta)
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
class OracleLatentT(nn.Module):
    def __init__(self, decode_fn):
        super().__init__(); self.enc=mlp([D,64,K_LAT]); self._decode=decode_fn
        self.M=nn.Parameter(torch.eye(K_LAT)+0.15*torch.randn(K_LAT,K_LAT))
    def forward(self,x): return self._decode(self.enc(x)@self.M.T)

MIN_MOVE=None   # set after estimating ~one-step move scale

def disc_recipe(loss,f,T,x):
    Tx=T(x)
    loss=loss+W_AUG*eqm_loss(f,Tx)
    move=(Tx-x).pow(2).sum(1).clamp_min(1e-8).sqrt()
    loss=loss+W_MOVE*torch.relu(MIN_MOVE-move).mean()         # hinge anti-identity
    xk=x
    for _ in range(P_CYCLE): xk=T(xk)
    loss=loss+W_CYC*(xk-x).pow(2).sum(1).mean()
    return loss

def train(arm, seed, steps, ae_steps, joint_mode):
    torch.manual_seed(seed)
    if DEV.type=="cuda": torch.cuda.manual_seed_all(seed)
    f=mlp([D+1,256,256,D]).to(DEV); T=None; aux=None
    if arm=="DISC_LINEAR":
        T=LinearT().to(DEV); opt=torch.optim.Adam(list(f.parameters())+list(T.parameters()),lr=2e-3,weight_decay=W_WD)
    elif arm=="ORACLE_LATENT":
        T=OracleLatentT(DECODE).to(DEV)
        eo=torch.optim.Adam(T.enc.parameters(),lr=2e-3)
        for _ in range(ae_steps):
            phi=sample_phi_train(BATCH); z=z_of(phi); x=DECODE(z+SIG*torch.randn_like(z)); eo.zero_grad()
            ((T.enc(x)-z)**2).mean().backward(); eo.step()
        with torch.no_grad():
            phi=sample_phi_train(2048); z=z_of(phi); x=DECODE(z+SIG*torch.randn_like(z))
            aux=float((T.enc(x)-z).norm(dim=1).mean()/z.norm(dim=1).mean())
        for p in T.enc.parameters(): p.requires_grad_(False)
        opt=torch.optim.Adam(list(f.parameters())+[T.M],lr=2e-3)
    elif arm=="DISC_JOINT":
        T=LatentT().to(DEV)
        if joint_mode=="warm":
            ao=torch.optim.Adam(list(T.enc.parameters())+list(T.dec.parameters()),lr=2e-3)
            for _ in range(ae_steps):
                x=sample_train(BATCH); ao.zero_grad(); ((T.recon(x)-x)**2).mean().backward(); ao.step()
        opt=torch.optim.Adam(list(f.parameters())+list(T.parameters()),lr=2e-3,weight_decay=W_WD)
    else:
        opt=torch.optim.Adam(f.parameters(),lr=2e-3)
    for step in range(steps):
        x=sample_train(BATCH)
        if arm=="DISC_JOINT" and joint_mode=="alt" and step%2==0:
            opt.zero_grad(); (W_RECON*((T.recon(x)-x)**2).mean()).backward(); opt.step(); continue
        opt.zero_grad(); loss=eqm_loss(f,x)
        if arm=="ORACLE":
            th=float(torch.rand(1))*2*math.pi                      # random rotation = full SO(2) group
            phi=sample_phi_train(BATCH); z=z_of(phi)@rot_mat(th).T
            loss=loss+W_AUG*eqm_loss(f,DECODE(z))
        elif arm in ("DISC_LINEAR","ORACLE_LATENT"):
            loss=disc_recipe(loss,f,T,x)
        elif arm=="DISC_JOINT":
            loss=disc_recipe(loss,f,T,x)+W_RECON*((T.recon(x)-x)**2).mean()
        loss.backward(); opt.step()
    if arm=="DISC_JOINT":
        with torch.no_grad(): x=sample_train(2048); aux=float((T.recon(x)-x).norm(dim=1).mean()/x.norm(dim=1).mean())
    return f,T,aux

@torch.no_grad()
def evaluate(f,T,aux):
    s=sample(f,N_SAMPLE); d=torch.cdist(s,MAN); nearest=d.argmin(1); within=d.min(1).values<3*SIG_OBS
    phi=PHI_GRID[nearest]
    rec=float((within & in_gap(phi)).float().sum()/N_SAMPLE)
    bins=torch.floor(phi/(2*math.pi)*36).clamp(0,35).long()
    cov=torch.zeros(36,device=DEV)
    for b in range(36): cov[b]=((bins==b)&within).float().sum()
    cov=cov/N_SAMPLE
    out={"recall_arc":rec,"arc_bins_covered":int((cov>0.005).sum()),"on_manifold_gen":float(within.float().mean())}
    if T is not None:
        xb=sample_train(512); Tx=T(xb)
        out["T_on_manifold"]=float((torch.cdist(Tx,MAN).min(1).values<3*SIG_OBS).float().mean())
        if hasattr(T,"M"):
            I=torch.eye(K_LAT,device=DEV); Mp=T.M.clone()
            for _ in range(P_CYCLE-1): Mp=Mp@T.M
            out["M_offidentity"]=float((T.M-I).norm()); out["M_order_err"]=float((Mp-I).norm())
            out["M_angle_deg"]=float(torch.atan2(T.M[1,0],T.M[0,0])*180/math.pi)
    if aux is not None: out["ae_diag"]=aux
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-json",default="results_rung7.json")
    ap.add_argument("--joint-mode",choices=["scratch","warm","alt"],default="scratch")
    ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()
    global MIN_MOVE
    # estimate "one-step" move = distance between manifold points ~ a sensible rotation step (45deg)
    with torch.no_grad():
        i0=0; i45=45; MIN_MOVE=float((MAN[i45]-MAN[i0]).norm())
    steps,ae_steps,seeds=(300,300,[0]) if args.quick else (STEPS,AE_STEPS,SEEDS)
    print(f"device={DEV} K_lat={K_LAT} held-out arc=[157.5,202.5]deg (45 wide) MIN_MOVE(45deg)={MIN_MOVE:.3f}")
    print(f"sig_obs={SIG_OBS:.3f}  ideal recall_arc={45/360:.3f}\n")
    results={"config":{"joint_mode":args.joint_mode,"steps":steps,"seeds":seeds,"device":str(DEV),"min_move":MIN_MOVE},"arms":{}}
    hdr=f"{'arm':14s} {'recall_arc':>10s} {'arcbins/36':>11s} {'onman_gen':>10s} {'AE/enc':>7s} {'M-I':>6s} {'M^8-I':>7s} {'Mang':>7s}"
    print(hdr); print("-"*len(hdr))
    for arm in ARMS:
        runs=[evaluate(*train(arm,s,steps,ae_steps,args.joint_mode)) for s in seeds]
        agg={}
        for k in ("recall_arc","arc_bins_covered","on_manifold_gen","T_on_manifold","ae_diag","M_offidentity","M_order_err","M_angle_deg"):
            v=[r[k] for r in runs if k in r]
            if v: agg[k]=sum(v)/len(v)
        agg["recall_sd"]=(sum((r["recall_arc"]-agg["recall_arc"])**2 for r in runs)/len(runs))**0.5
        results["arms"][arm]=agg
        def gg(k,fmt="%.3f",w=7): return (fmt%agg[k]).rjust(w) if k in agg else "   -  ".rjust(w)
        print(f"{arm:14s} {agg['recall_arc']:6.3f}±{agg['recall_sd']:.3f} {agg['arc_bins_covered']:11.1f} "
              f"{gg('on_manifold_gen',w=10)} {gg('ae_diag')} {gg('M_offidentity','%.2f',6)} {gg('M_order_err','%.3f',7)} {gg('M_angle_deg','%.1f',7)}")
    print(f"\nideal recall_arc={45/360:.3f}  (M_angle ~ multiple of 45 + recall>0 => rotation discovered)")
    os.makedirs(os.path.dirname(args.out_json) or ".",exist_ok=True)
    json.dump(results,open(args.out_json,"w"),indent=2)
    print(f"\nwrote {args.out_json}")

if __name__=="__main__": main()
