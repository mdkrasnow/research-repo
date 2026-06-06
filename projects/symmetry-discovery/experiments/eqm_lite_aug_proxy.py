"""v14 Rung E — EqM-lite augmentation proxy (closest cheap analog of the real target).

Tiny EqM/flow-shaped field-matching model + the v13/v14 augmentation loss:
    loss = eqm_loss(x) + lambda * eqm_loss(T(x))
Train a small field model on a CIFAR subset under each augmentation; evaluate the field error on
TRANSLATED validation views (translation-robustness of the learned field — the property crop aug buys).

Arms (same as Rung D): base, known_crop (U[-4,4]^2), single (v13 one fixed shift),
random_dist (N(0,sd^2)), disc_dist (N(discovered mean, std^2)). Discovery is unsupervised (_se2_discovery).

PASS: disc_dist translated-EqM-loss < single AND < random_dist, and closes a meaningful fraction of the
(base - known_crop) robustness gap (lower loss = better).
"""
import argparse, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from _se2_discovery import discover_single, discover_distribution, warp_txty

DEV = torch.device("cpu")


class TinyEqM(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(4, 48, 3, 1, 1); self.c2 = nn.Conv2d(48, 48, 3, 1, 1); self.c3 = nn.Conv2d(48, 3, 3, 1, 1)
    def forward(self, x, g):
        gch = g.view(-1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        h = F.relu(self.c1(torch.cat([x, gch], 1))); h = F.relu(self.c2(h))
        return self.c3(h)


def eqm_loss(model, x, draws=1):
    tot = 0.0
    for _ in range(draws):
        eps = torch.randn_like(x); g = torch.rand(x.size(0), device=x.device)
        gg = g.view(-1, 1, 1, 1)
        xg = (1 - gg) * x + gg * eps           # flow interpolation data->noise
        target = eps - x                        # velocity field
        tot = tot + F.mse_loss(model(xg, g), target)
    return tot / draws


def aug_fn(name, single_txty, dmean, dstd, sd_rand=4.0):
    def f(x):
        B = x.size(0)
        if name == "base": return x
        if name == "known_crop": tt = (torch.rand(B, 2, device=x.device) * 2 - 1) * 4.0
        elif name == "single": tt = single_txty.to(x.device).unsqueeze(0).repeat(B, 1)
        elif name == "random_dist": tt = torch.randn(B, 2, device=x.device) * sd_rand
        else: tt = dmean.to(x.device) + torch.randn(B, 2, device=x.device) * dstd.to(x.device)
        return warp_txty(x, tt)
    return f


@torch.no_grad()
def eval_translated(model, xva, draws=8):
    tt = (torch.rand(xva.size(0), 2) * 2 - 1) * 4.0
    return float(eqm_loss(model, warp_txty(xva, tt), draws=draws))


def train_arm(xtr, xva, aug, epochs, lam=0.3, bs=128, seed=0):
    torch.manual_seed(seed); net = TinyEqM().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = xtr.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            xb = xtr[perm[i:i+bs]]
            loss = eqm_loss(net, xb) + lam * eqm_loss(net, aug(xb))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        clean = float(eqm_loss(net, xva, draws=8))
    return clean, eval_translated(net, xva)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_eqm_lite.json")
    ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    epochs = 3 if a.quick else 6; disc_steps = 250 if a.quick else 500
    n_seeds = 1 if a.quick else 3
    torch.manual_seed(0)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    root = next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data"))
                 if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))), "data")
    ds = datasets.CIFAR10(root, train=True, download=False, transform=tf)
    idx = torch.randperm(len(ds))[:4000].tolist()
    X = torch.stack([ds[i][0] for i in idx])
    xtr, xva = X[:3000], X[3000:]

    print("discovering operators (unsupervised)...")
    single = discover_single(xtr.clone(), steps=disc_steps)
    dmean, dstd = discover_distribution(xtr.clone(), steps=disc_steps)
    print(f"  single=({single[0]:.2f},{single[1]:.2f}) dist mean=({dmean[0]:.2f},{dmean[1]:.2f}) std=({dstd[0]:.2f},{dstd[1]:.2f})px")

    arms = ["base", "known_crop", "single", "random_dist", "disc_dist"]
    res = {"single_txty": single.tolist(), "disc_mean": dmean.tolist(), "disc_std": dstd.tolist(), "n_seeds": n_seeds, "arms": {}}
    print(f"\n{'arm':14s} {'clean_L':>8s} {'transl_L':>9s} {'transl_std':>10s}  (lower=better, mean/{n_seeds} seeds)")
    for name in arms:
        aug = aug_fn(name, single, dmean, dstd)
        cs, ts = [], []
        for sd in range(n_seeds):
            c, t = train_arm(xtr, xva, aug, epochs, seed=sd); cs.append(c); ts.append(t)
        clean = sum(cs)/len(cs); transl = sum(ts)/len(ts)
        tstd = (sum((t-transl)**2 for t in ts)/len(ts))**0.5
        res["arms"][name] = {"clean_loss": clean, "translated_loss": transl, "translated_std": tstd, "seeds": ts}
        print(f"{name:14s} {clean:8.4f} {transl:9.4f} {tstd:10.4f}")

    A = res["arms"]
    base_t = A["base"]["translated_loss"]; known_t = A["known_crop"]["translated_loss"]
    disc_t = A["disc_dist"]["translated_loss"]; single_t = A["single"]["translated_loss"]; rand_t = A["random_dist"]["translated_loss"]
    gap = base_t - known_t                     # known should LOWER translated loss
    closed = (base_t - disc_t) / gap if gap > 1e-6 else 0.0
    beats_single = disc_t < single_t; beats_rand = disc_t < rand_t; known_helps = known_t < base_t - 1e-4
    # signal guard: if the known-crop gap is within noise, the proxy cannot distinguish arms -> INCONCLUSIVE
    noise = max(A["base"]["translated_std"], A["known_crop"]["translated_std"], A["disc_dist"]["translated_std"])
    has_signal = gap > 2 * noise and gap > 0.003
    print(f"\nknown-crop gap (base_L - known_L) = {gap:.4f}  (noise~{noise:.4f}; signal={has_signal})")
    print(f"disc closes {closed*100:.0f}% | disc<single: {beats_single} | disc<random_dist: {beats_rand} | known<base: {known_helps}")
    if not has_signal:
        print("RUNG E: INCONCLUSIVE — EqM-lite translated-field gap is within noise; proxy can't distinguish augs")
        ok = None
    else:
        ok = known_helps and beats_single and beats_rand and closed > 0.3
        print("RUNG E:", "PASS — discovered distribution beats single+random on the EqM-lite translated field"
              if ok else "FAIL — discovered distribution does not beat random / closes <30% (interpret)")
    json.dump({**res, "verdict": ("inconclusive" if ok is None else ("pass" if ok else "fail"))}, open(a.out_json, "w"), indent=2)
    print(f"wrote {a.out_json}")
    return ok


if __name__ == "__main__":
    v = main()
    raise SystemExit(0 if v else 1)
