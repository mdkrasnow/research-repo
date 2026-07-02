"""v16 Exp 3 — stage-conditioned residual CURRICULUM (safe adaptivity: timing, not per-image).

Exp 2: a mild residual on top of BEST_KNOWN marginally helps, but LEARNING its shape == RANDOM residual.
Exp 3 tests an ORTHOGONAL lever: maybe the residual SHAPE has no signal but its TIMING does — apply weak
residual early, stronger later (or vice-versa). strength(step_frac) = sigmoid(a + b*step_frac) scales a FIXED
mild residual range. This is adaptivity WITHOUT per-image adversarial degeneracy (v15's failure mode).

φ=(a,b) optimized by ES on the SAME short-run validation utility (anchor/entropy not needed — ranges fixed,
strength∈[0,1] is always on-manifold). Arms: best_known, static_residual (strength≡1), curriculum_learned
(ES φ), random_curriculum (random φ). 3 seeds, separate test split.
PASS: curriculum beats static AND best_known beyond noise. Else: timing adds nothing -> v16 STOP (no Exp4).
"""
import argparse, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from v16_known_aug_ceiling import warp, color_jitter, make_aug, Net, evaluate

DEV = torch.device("cpu")
FIXED_R = {"r_tx": 1.5, "r_ty": 1.5, "r_s": 0.07}  # fixed mild residual range (color OFF; best_known has it)


def strength(phi, frac):
    return float(torch.sigmoid(phi[0] + phi[1]*frac))


def residual_scaled(x, base_fn, s):
    xb = base_fn(x); B = x.size(0)
    tx = (torch.rand(B, device=x.device)*2-1)*FIXED_R["r_tx"]*s
    ty = (torch.rand(B, device=x.device)*2-1)*FIXED_R["r_ty"]*s
    sc = 1 + (torch.rand(B, device=x.device)*2-1)*FIXED_R["r_s"]*s
    return warp(xb, torch.stack([tx, ty], 1), sc)


def inner_train_util(phi, base_fn, xtr, ytr, xpv, ypv, K=300, bs=128, fixed_seed=1234):
    torch.manual_seed(fixed_seed); net = Net().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = xtr.size(0); step = 0
    while step < K:
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            if step >= K: break
            s = strength(phi, step/K); idx = perm[i:i+bs]
            loss = F.cross_entropy(net(residual_scaled(xtr[idx], base_fn, s)), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
    return evaluate(net, xpv, ypv, "robust")


def es_optimize(base_fn, xtr, ytr, xpv, ypv, K, P, O, sigma=0.5, lr=0.5, seed=0):
    torch.manual_seed(seed); phi = torch.zeros(2)  # init: const strength sigmoid(0)=0.5
    for o in range(O):
        eps = torch.randn(P, 2)*sigma
        us = torch.tensor([inner_train_util(phi+eps[p], base_fn, xtr, ytr, xpv, ypv, K=K) for p in range(P)])
        adv = (us - us.mean())/(us.std()+1e-6); phi = phi + lr*(adv.unsqueeze(1)*eps).mean(0)
        print(f"  ES o{o}: U~{float(us.mean()):.3f} phi={[round(float(x),3) for x in phi]} "
              f"s(0)={strength(phi,0):.2f} s(1)={strength(phi,1):.2f}")
    return phi.detach()


def final_eval(aug_step_fn, xtr, ytr, xte, yte, epochs, n_seeds, bs=128):
    cs, rs = [], []
    for sd in range(n_seeds):
        torch.manual_seed(sd); net = Net().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        n = xtr.size(0); total = epochs*((n+bs-1)//bs); step = 0
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]; xb = aug_step_fn(xtr[idx], step/max(1,total)); step += 1
                loss = F.cross_entropy(net(xb), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
        cs.append(evaluate(net, xte, yte, "clean")); rs.append(evaluate(net, xte, yte, "robust"))
    rob = sum(rs)/len(rs); return {"clean": sum(cs)/len(cs), "robust": rob,
                                   "robust_std": (sum((r-rob)**2 for r in rs)/len(rs))**0.5, "seeds": rs}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_v16_exp3.json")
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--best-known", default="crop_pad4_color")
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
    base_fn = make_aug(a.best_known)

    print(f"BEST_KNOWN={a.best_known}; ES optimizing curriculum schedule φ=(a,b)...")
    phi_cur = es_optimize(base_fn, xtr, ytr, xpv, ypv, K, P, O, seed=1)
    torch.manual_seed(7); phi_rand = torch.randn(2)*1.0

    augs = {
        "best_known": lambda x, frac: base_fn(x),
        "static_residual": lambda x, frac: residual_scaled(x, base_fn, 1.0),
        "curriculum_learned": lambda x, frac: residual_scaled(x, base_fn, strength(phi_cur, frac)),
        "random_curriculum": lambda x, frac: residual_scaled(x, base_fn, strength(phi_rand, frac)),
    }
    res = {"best_known": a.best_known, "n_seeds": n_seeds,
           "phi_curriculum": phi_cur.tolist(), "phi_random": phi_rand.tolist(),
           "s_start": strength(phi_cur,0), "s_end": strength(phi_cur,1), "arms": {}}
    print(f"\ncurriculum schedule: s(0)={strength(phi_cur,0):.2f} -> s(1)={strength(phi_cur,1):.2f}")
    print(f"{'arm':20s} {'clean':>6s} {'robust':>7s} {'rstd':>6s}")
    for name, fn in augs.items():
        d = final_eval(fn, xtr, ytr, xte, yte, epochs, n_seeds); res["arms"][name] = d
        print(f"{name:20s} {d['clean']:6.3f} {d['robust']:7.3f} {d['robust_std']:6.3f}")

    A = res["arms"]; bk = A["best_known"]["robust"]; st = A["static_residual"]["robust"]
    cur = A["curriculum_learned"]["robust"]; rnd = A["random_curriculum"]["robust"]
    noise = max(A["curriculum_learned"]["robust_std"], A["static_residual"]["robust_std"], A["best_known"]["robust_std"])
    beats_static = cur > st + max(1e-4, 0.5*noise); beats_bk = cur > bk + noise; beats_randcur = cur > rnd + max(1e-4,0.5*noise)
    res.update({"beats_static": beats_static, "beats_best_known": beats_bk, "beats_random_curriculum": beats_randcur})
    print(f"\ncurriculum={cur:.4f} static={st:.4f} best_known={bk:.4f} random_cur={rnd:.4f} noise~{noise:.4f}")
    print(f"beats_static:{beats_static} beats_best_known:{beats_bk} beats_random_curriculum:{beats_randcur}")
    ok = beats_static and beats_bk and beats_randcur
    if ok:
        print("RUNG E3: PASS — curriculum (timing) BEATS static + best-known + random schedule. Proceed to Exp 4 (EqM-lite).")
    else:
        print("RUNG E3: NO BEAT — timing adds nothing over static/random. v16 STOP: residual SHAPE (E2) and TIMING "
              "(E3) both == random over a known generic base. Do not run Exp 4 / FID.")
    json.dump({**res, "verdict": "pass" if ok else "no_beat"}, open(a.out_json,"w"), indent=2)
    print(f"wrote {a.out_json}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
