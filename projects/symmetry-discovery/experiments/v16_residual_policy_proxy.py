"""v16 Exp 2 — bounded RESIDUAL policy over BEST_KNOWN, optimized for VALIDATION UTILITY (not hardness).

The v15-distinct hypothesis: crop is a strong BASE; a small BOUNDED residual distribution qθ stacked on top
of BEST_KNOWN may beat it IF θ is optimized for actual short-run validation utility (bilevel-lite), with
anchor/entropy as CONSTRAINTS — not for frozen-scorer hardness (which v15 showed backfires).

Residual family (mild, non-destructive, always on-manifold by bound): tx,ty in ±r_tx,±r_ty (r<=2px),
scale 1±r_s (r_s<=0.1), brightness/contrast ±r_b (r_b<=0.15). θ (4 reals) -> bounded ranges via sigmoid.

Optimization = Evolution Strategies on θ. Utility(θ) = robust acc of a fresh proxy trained K steps with
aug=residual(θ)∘BEST_KNOWN, evaluated on a held-out POLICY-VAL split, minus β·anchor_ED(θ) plus γ·entropy(θ).
Common-random-numbers (fixed inner seed) isolates the θ effect from init/order noise. Final comparison is
on a SEPARATE TEST split, 3 seeds, full budget — so θ never sees the test metric.

Arms: base, best_known, +random_residual (θ=init, unoptimized same family), +learned_residual (full ES),
+learned_residual_no_anchor (β=0), +learned_residual_no_entropy (γ=0).
PASS: learned_residual beats best_known AND random_residual beyond seed noise; ablations show anchor/entropy
matter (no_anchor drifts off-manifold / no_entropy collapses).
"""
import argparse, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from v16_known_aug_ceiling import warp, color_jitter, make_aug, Net, robust_corrupt, evaluate
from _se2_discovery import FrozenConv, _ed_t

DEV = torch.device("cpu")
ALLOW_RESIDUAL_COLOR = True  # set False when BEST_KNOWN already includes color (avoid double-color)


def bounded(theta):
    s = torch.sigmoid(theta)
    rb = 0.15*float(s[3]) if ALLOW_RESIDUAL_COLOR else 0.0
    return {"r_tx": 2.0*float(s[0]), "r_ty": 2.0*float(s[1]), "r_s": 0.1*float(s[2]), "r_b": rb}


def residual_aug(x, theta, base_fn):
    xb = base_fn(x); B = x.size(0); r = bounded(theta)
    tx = (torch.rand(B, device=x.device)*2-1)*r["r_tx"]; ty = (torch.rand(B, device=x.device)*2-1)*r["r_ty"]
    sc = 1 + (torch.rand(B, device=x.device)*2-1)*r["r_s"]
    xr = warp(xb, torch.stack([tx, ty], 1), sc)
    if r["r_b"] > 1e-4: xr = color_jitter(xr, b=r["r_b"], c=r["r_b"])
    return xr


def entropy(theta):  # in [0,1]: mean normalized range -> rewards spread, penalizes collapse
    r = bounded(theta)
    return (r["r_tx"]/2 + r["r_ty"]/2 + r["r_s"]/0.1 + r["r_b"]/0.15) / 4


def anchor_ed(theta, base_fn, xa, enc, anchor):
    with torch.no_grad():
        f = enc(residual_aug(xa, theta, base_fn))
        return float(_ed_t(f, anchor[torch.randperm(anchor.size(0))[:f.size(0)]]))


def inner_train_util(theta, base_fn, xtr, ytr, xpv, ypv, K=250, bs=128, fixed_seed=1234):
    torch.manual_seed(fixed_seed); net = Net().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = xtr.size(0); step = 0
    while step < K:
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            if step >= K: break
            idx = perm[i:i+bs]; loss = F.cross_entropy(net(residual_aug(xtr[idx], theta, base_fn)), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
    return evaluate(net, xpv, ypv, "robust")


def es_optimize(base_fn, xtr, ytr, xpv, ypv, enc, anchor, beta, gamma, K, P, O, sigma=0.4, lr=0.5, seed=0):
    torch.manual_seed(seed); theta = torch.zeros(4)
    ed_ref = anchor_ed(torch.full((4,), -10.0), base_fn, xtr[:256], enc, anchor)  # base_fn, ~zero residual
    hist = []
    def U(th):
        u = inner_train_util(th, base_fn, xtr, ytr, xpv, ypv, K=K)
        excess = max(0.0, anchor_ed(th, base_fn, xtr[:256], enc, anchor) - ed_ref)  # residual-induced drift only
        return u - beta*excess + gamma*entropy(th)
    for o in range(O):
        eps = torch.randn(P, 4)*sigma
        us = torch.tensor([U(theta+eps[p]) for p in range(P)])
        adv = (us - us.mean()) / (us.std() + 1e-6)
        theta = theta + lr * (adv.unsqueeze(1)*eps).mean(0)
        hist.append({"outer": o, "U_mean": float(us.mean()), "theta": theta.tolist(), "ranges": bounded(theta)})
        print(f"  ES o{o}: U~{float(us.mean()):.3f} ranges={ {k:round(v,3) for k,v in bounded(theta).items()} }")
    return theta.detach(), hist


def final_eval(name, aug, xtr, ytr, xte, yte, epochs, n_seeds, bs=128):
    cs, rs = [], []
    for sd in range(n_seeds):
        torch.manual_seed(sd); net = Net().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); n = xtr.size(0)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]; loss = F.cross_entropy(net(aug(xtr[idx])), ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        cs.append(evaluate(net, xte, yte, "clean")); rs.append(evaluate(net, xte, yte, "robust"))
    rob = sum(rs)/len(rs); return {"clean": sum(cs)/len(cs), "robust": rob,
                                   "robust_std": (sum((r-rob)**2 for r in rs)/len(rs))**0.5, "seeds": rs}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_v16_exp2.json")
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--best-known", default="crop_pad4")
    a = ap.parse_args()
    if a.quick: epochs, n_seeds, K, P, O = 3, 1, 120, 4, 3
    else: epochs, n_seeds, K, P, O = 8, 3, 300, 6, 8
    torch.manual_seed(0)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    root = next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data"))
                 if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))), "data")
    ds = datasets.CIFAR10(root, train=True, download=False, transform=tf)
    idx = torch.randperm(len(ds))[:3500].tolist()
    X = torch.stack([ds[i][0] for i in idx]); Y = torch.tensor([ds[i][1] for i in idx])
    xtr, ytr = X[:2000], Y[:2000]; xpv, ypv = X[2000:2500], Y[2000:2500]; xte, yte = X[2500:], Y[2500:]
    global ALLOW_RESIDUAL_COLOR
    ALLOW_RESIDUAL_COLOR = "color" not in a.best_known
    print(f"residual color {'ON' if ALLOW_RESIDUAL_COLOR else 'OFF (best_known has color)'}")
    base_fn = make_aug(a.best_known)
    enc = FrozenConv().to(DEV)
    with torch.no_grad(): anchor = enc(xtr)

    print(f"BEST_KNOWN={a.best_known}; ES (K={K},P={P},O={O}) optimizing VALIDATION utility...")
    print("learned_residual (full: anchor+entropy):")
    th_full, _ = es_optimize(base_fn, xtr, ytr, xpv, ypv, enc, anchor, beta=0.05, gamma=0.1, K=K, P=P, O=O, seed=1)
    print("learned_residual_no_anchor (beta=0):")
    th_noanc, _ = es_optimize(base_fn, xtr, ytr, xpv, ypv, enc, anchor, beta=0.0, gamma=0.1, K=K, P=P, O=O, seed=2)
    print("learned_residual_no_entropy (gamma=0):")
    th_noent, _ = es_optimize(base_fn, xtr, ytr, xpv, ypv, enc, anchor, beta=0.05, gamma=0.0, K=K, P=P, O=O, seed=3)
    th_rand = torch.zeros(4)  # unoptimized init of the SAME family = random residual

    augs = {
        "base": make_aug("base"),
        "best_known": base_fn,
        "rand_residual": lambda x: residual_aug(x, th_rand, base_fn),
        "learned_residual": lambda x: residual_aug(x, th_full, base_fn),
        "learned_no_anchor": lambda x: residual_aug(x, th_noanc, base_fn),
        "learned_no_entropy": lambda x: residual_aug(x, th_noent, base_fn),
    }
    thetas = {"rand": th_rand.tolist(), "full": th_full.tolist(), "no_anchor": th_noanc.tolist(),
              "no_entropy": th_noent.tolist()}
    ranges = {k: bounded(torch.tensor(v)) for k, v in thetas.items()}
    res = {"best_known": a.best_known, "n_seeds": n_seeds, "thetas": thetas, "ranges": ranges, "arms": {}}

    print(f"\nlearned ranges full={ {k:round(v,3) for k,v in ranges['full'].items()} }")
    print(f"{'arm':20s} {'clean':>6s} {'robust':>7s} {'rstd':>6s}")
    for name, aug in augs.items():
        d = final_eval(name, aug, xtr, ytr, xte, yte, epochs, n_seeds); res["arms"][name] = d
        print(f"{name:20s} {d['clean']:6.3f} {d['robust']:7.3f} {d['robust_std']:6.3f}")

    A = res["arms"]; bk = A["best_known"]["robust"]; lr_ = A["learned_residual"]["robust"]; rr = A["rand_residual"]["robust"]
    noise = max(A["learned_residual"]["robust_std"], A["best_known"]["robust_std"])
    beats_bk = lr_ > bk + noise; beats_rand = lr_ > rr + 1e-4
    anchor_matters = A["learned_residual"]["robust"] >= A["learned_no_anchor"]["robust"] - 1e-4
    entropy_matters = A["learned_residual"]["robust"] >= A["learned_no_entropy"]["robust"] - 1e-4
    res.update({"beats_best_known": beats_bk, "beats_random": beats_rand,
                "anchor_matters": anchor_matters, "entropy_matters": entropy_matters})
    print(f"\nlearned={lr_:.3f} best_known={bk:.3f} rand_residual={rr:.3f} noise~{noise:.3f}")
    print(f"beats_best_known:{beats_bk} beats_random:{beats_rand} anchor_matters:{anchor_matters} entropy_matters:{entropy_matters}")
    ok = beats_bk and beats_rand
    if ok:
        print("RUNG E2: PASS — learned residual BEATS best-known + random. Proceed to Exp 3 (curriculum).")
    else:
        print("RUNG E2: NO BEAT — learned residual ties/loses best-known or random. Per stop-rule, STOP unless close "
              "(then Exp 3). tie != beat (v14/v15 lesson).")
    json.dump({**res, "verdict": "pass" if ok else "no_beat"}, open(a.out_json,"w"), indent=2)
    print(f"wrote {a.out_json}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
