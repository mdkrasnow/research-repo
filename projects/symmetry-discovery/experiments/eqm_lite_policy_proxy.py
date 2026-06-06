"""v14 (beat-crop) Rung D — EqM-lite policy proxy. loss = eqm_loss(x) + lam * E_{T~q}[eqm_loss(T(x))].

Arms: BASE, KNOWN_CROP (U[-4,4]^2), RANDOM_POLICY, DISCOVERED_POLICY (anchor+entropy), SINGLE_DISCOVERED.
Eval: clean + BROAD translated (±6px) EqM field loss (lower=more translation-robust field). Signal-guard:
if the known-crop gap is within noise the proxy can't distinguish augs -> INCONCLUSIVE (the velocity field
eps-x is locally translation-robust, so this proxy was low-signal before).
PASS: DISCOVERED_POLICY beats KNOWN_CROP or clearly closes the gap AND beats random + single.
"""
import argparse, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from _se2_discovery import discover_policy, discover_single, policy_txty, warp_txty

DEV = torch.device("cpu")


class TinyEqM(nn.Module):
    def __init__(self):
        super().__init__(); self.c1=nn.Conv2d(4,48,3,1,1); self.c2=nn.Conv2d(48,48,3,1,1); self.c3=nn.Conv2d(48,3,3,1,1)
    def forward(self, x, g):
        gch = g.view(-1,1,1,1).expand(x.size(0),1,x.size(2),x.size(3))
        h = F.relu(self.c1(torch.cat([x,gch],1))); h = F.relu(self.c2(h)); return self.c3(h)


def eqm_loss(model, x, draws=1):
    tot = torch.zeros((), device=x.device)
    for _ in range(draws):
        eps = torch.randn_like(x); g = torch.rand(x.size(0), device=x.device); gg = g.view(-1,1,1,1)
        tot = tot + F.mse_loss(model((1-gg)*x+gg*eps, g), eps-x)
    return tot/draws


def aug_fn(name, pol, single, rand_pol):
    def f(x):
        B = x.size(0)
        if name=="base": return x
        if name=="known_crop": return warp_txty(x, (torch.rand(B,2,device=x.device)*2-1)*4.0)
        if name=="random_policy": return warp_txty(x, policy_txty(*rand_pol, B, x.device))
        if name=="single": return warp_txty(x, single.to(x.device).unsqueeze(0).repeat(B,1))
        return warp_txty(x, policy_txty(*pol, B, x.device))
    return f


def train_arm(xtr, xva, aug, epochs, lam=0.3, bs=128, seed=0):
    torch.manual_seed(seed); net = TinyEqM().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); n=xtr.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            xb = xtr[perm[i:i+bs]]; loss = eqm_loss(net, xb) + lam*eqm_loss(net, aug(xb))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        clean = float(eqm_loss(net, xva, draws=8))
        xt = warp_txty(xva, (torch.rand(xva.size(0),2)*2-1)*6.0)
        transl = float(eqm_loss(net, xt, draws=8))
    return clean, transl


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-json",default="results_eqm_lite_policy.json")
    ap.add_argument("--quick",action="store_true"); a=ap.parse_args()
    epochs=3 if a.quick else 6; dsteps=200 if a.quick else 450; n_seeds=1 if a.quick else 3
    torch.manual_seed(0)
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,)*3,(0.5,)*3)])
    root=next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data")) if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))),"data")
    ds=datasets.CIFAR10(root,train=True,download=False,transform=tf)
    X=torch.stack([ds[i][0] for i in torch.randperm(len(ds))[:4000].tolist()]); xtr,xva=X[:3000],X[3000:]
    print("discovering policy + single (unsupervised)...")
    pol=discover_policy(xtr.clone(), steps=dsteps, entropy=True)
    single=discover_single(xtr.clone(), steps=dsteps)
    torch.manual_seed(99); rand_pol=(0.15*torch.randn(2,3),0.15*torch.randn(2,3),torch.tensor([0.0,0.0]))

    arms=["base","known_crop","random_policy","single","disc_policy"]
    res={"n_seeds":n_seeds,"arms":{}}
    print(f"\n{'arm':14s} {'clean_L':>8s} {'transl_L':>9s} {'tstd':>7s}")
    for name in arms:
        aug=aug_fn(name,pol,single,rand_pol); cs,ts=[],[]
        for sd in range(n_seeds):
            c,t=train_arm(xtr,xva,aug,epochs,seed=sd); cs.append(c); ts.append(t)
        clean=sum(cs)/len(cs); transl=sum(ts)/len(ts); tstd=(sum((t-transl)**2 for t in ts)/len(ts))**0.5
        res["arms"][name]={"clean_loss":clean,"translated_loss":transl,"translated_std":tstd,"seeds":ts}
        print(f"{name:14s} {clean:8.4f} {transl:9.4f} {tstd:7.4f}")
    A=res["arms"]; base=A["base"]["translated_loss"]; crop=A["known_crop"]["translated_loss"]
    disc=A["disc_policy"]["translated_loss"]; rand=A["random_policy"]["translated_loss"]; sing=A["single"]["translated_loss"]
    noise=max(A["base"]["translated_std"],A["known_crop"]["translated_std"],A["disc_policy"]["translated_std"])
    gap=base-crop; has_signal = gap>2*noise and gap>0.003
    print(f"\nknown gap (base-crop)={gap:.4f} noise~{noise:.4f} signal={has_signal}")
    if not has_signal:
        print("RUNG D: INCONCLUSIVE — EqM-lite translated-field gap within noise (velocity field locally translation-robust)")
        verdict="inconclusive"; ok=None
    else:
        ok = (disc<crop or (base-disc)/gap>0.8) and disc<rand and disc<sing
        verdict="pass" if ok else "fail"
        print(f"disc<crop:{disc<crop} closes{(base-disc)/gap*100:.0f}% disc<random:{disc<rand} disc<single:{disc<sing}")
        print("RUNG D:", "PASS — discovered policy beats/matches crop on EqM-lite + beats random/single" if ok else "FAIL")
    json.dump({**res,"verdict":verdict},open(a.out_json,"w"),indent=2); print(f"wrote {a.out_json}")
    return ok


if __name__=="__main__":
    v=main(); raise SystemExit(0 if v else 1)
