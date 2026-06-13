"""D2 step 4a — pre-flight gate per candidate rung (preregistration-d2.md).

Per candidate from d2_knob_search.json:
  H = mean(supcon) - mean(no_mine), 10 seeds each (training, cheap)
  noise floor = SD(no_mine across seeds)
  S = certified-neg fraction of top similarity decile at lam* (alpha_0)
  rho_tail re-measured; gamma recorded if instrumented.
Gate: (a) H > 2x floor  (b) S > 0.10  (c) |rho_tail - target| <= 0.05.
Fail (a) or (b) -> rung trains only {no-mine (have), naive_neg} downstream.

Also: semi-real CIFAR rung (built + pre-flighted here; encoder = rn18
ImageNet, label-free w.r.t. CIFAR per A1' guard).

Writes results/d2_preflight.json (rung table + designated-rung selection).
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm import ladder                                          # noqa: E402
from rc_hpm.ladder import (Rung, rho_tail, calibrate_rung, supply_S,
                           gamma_probe, teacher_embed, draw,
                           train_arm)                              # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
H_BINS = [("low", 0.0, 0.02), ("med", 0.02, 0.06), ("high", 0.06, 9.9)]
N_SEEDS = 10


def _arm_job(args):
    kind, K, sigma, a, arm, seed = args
    if kind == "cifar":
        cifar_patch()                      # workers re-import: patch locally
        rung = make_cifar_rung()
    else:
        rung = Rung(K=K, sigma=sigma, a=a)
    return train_arm(rung, arm, seed)


# --------------------------- semi-real CIFAR rung ---------------------------

CIFAR_NPZ = os.path.join(RESULTS, "cifar_rn18_embeddings.npz")


def build_cifar_embeddings():
    if os.path.exists(CIFAR_NPZ):
        return
    import torch
    import torchvision
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = torchvision.datasets.CIFAR10(
        os.path.join(RESULTS, "..", "data"), train=True, transform=tf)
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Identity(); net.eval()
    idx = np.random.default_rng(0).choice(len(ds), 20_000, replace=False)
    feats, labs = [], []
    import torch as th
    with th.no_grad():
        for s0 in range(0, len(idx), 256):
            batch = th.stack([ds[int(i)][0] for i in idx[s0:s0 + 256]])
            feats.append(net(batch).numpy())
            labs.append(np.array([ds.targets[int(i)] for i in idx[s0:s0 + 256]]))
    emb = np.concatenate(feats)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)
    np.savez(CIFAR_NPZ, emb=emb, labels=np.concatenate(labs))


class CifarRung:
    """Pool-backed rung: x = frozen rn18 embedding (512-d); teacher = identity.
    First 16k = train pool; last 4k = probe holdout (fresh-draw substitute)."""
    K = 10
    dim = 512
    sigma = None
    a = None

    def __init__(self):
        z = np.load(CIFAR_NPZ)
        self.emb, self.labels = z["emb"], z["labels"]
        self.split = 16_000
        # per-dim SD for the embedding-space augmentation (prereg: 0.1 x SD)
        self.aug_sd = 0.1 * self.emb[:self.split].std(0)

    def tag(self):
        return "cifar_rn18"


def cifar_draw(rung: CifarRung, n, rng, pool="train"):
    lo, hi = (0, rung.split) if pool == "train" else (rung.split,
                                                     len(rung.labels))
    idx = rng.integers(lo, hi, n)
    return rung.emb[idx], rung.labels[idx]


def make_cifar_rung():
    build_cifar_embeddings()
    r = CifarRung()
    # monkey-route the ladder module helpers for this rung type
    return r


def cifar_patch():
    """Route ladder.draw/teacher_embed/aug for CifarRung instances."""
    base_draw = ladder.draw
    base_te = ladder.teacher_embed
    import rc_hpm.toy as toy
    base_aug = toy.aug_view

    def draw2(rung, n, rng):
        if isinstance(rung, CifarRung):
            return cifar_draw(rung, n, rng)
        return base_draw(rung, n, rng)

    def te2(rung, x):
        if isinstance(rung, CifarRung):
            return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
        return base_te(rung, x)

    ladder.draw = draw2
    ladder.teacher_embed = te2

    base_probe = ladder.linear_probe

    def probe2(student, rung, rng, **kw):
        if isinstance(rung, CifarRung):
            # probe data from the HOLDOUT pool (fresh-draw substitute)
            import torch
            xtr, ytr = cifar_draw(rung, 1000, rng, pool="hold")
            xte, yte = cifar_draw(rung, 2000, rng, pool="hold")
            with torch.no_grad():
                ztr = student(torch.tensor(xtr, dtype=torch.float32))
                zte = student(torch.tensor(xte, dtype=torch.float32))
            Wp = torch.zeros(ztr.shape[1], rung.K, requires_grad=True)
            bp = torch.zeros(rung.K, requires_grad=True)
            opt = torch.optim.Adam([Wp, bp], lr=0.05)
            Ytr = torch.tensor(ytr)
            for _ in range(300):
                loss = torch.nn.functional.cross_entropy(ztr @ Wp + bp, Ytr)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                return float(((zte @ Wp + bp).argmax(1).numpy() == yte).mean())
        return base_probe(student, rung, rng, **kw)

    ladder.linear_probe = probe2
    # aug law for the embedding rung: toy.aug_view(x, rng, 0.25) on unit-norm
    # coords — pinned as this rung's augmentation (documented).
    # P1(b) note: calibration fold (32k) resamples a 16k pool -> guarantee is
    # w.r.t. the pool empirical distribution (reported, not hidden).
    return


# ------------------------------- pre-flight ---------------------------------

def preflight_rung(spec, gamma_ok):
    kind = spec.get("kind", "toy")
    if kind == "cifar":
        rung = make_cifar_rung()
        rt_target = None
        # rho_tail on the pool
        rng = np.random.default_rng(0)
        x, lab = cifar_draw(rung, 20_000, rng)
        e = x
        i = rng.integers(0, len(x), 200_000)
        j = rng.integers(0, len(x), 200_000)
        keep = i != j; i, j = i[keep], j[keep]
        s = (e[i] * e[j]).sum(1)
        top = s >= np.quantile(s, 0.9)
        rt = float((lab[i] == lab[j])[top].mean())
    else:
        rung = Rung(K=spec["K"], sigma=spec["sigma"], a=spec["a"])
        rt_target = spec["target"]
        rt = rho_tail(rung)

    q_fn, calib = (None, None)
    try:
        q_fn, calib = calibrate_rung(rung)
        S = supply_S(rung, q_fn, calib)
    except Exception as e:                                # noqa: BLE001
        S = 0.0
        calib = None
        print("calibration error:", rung.tag(), e, flush=True)

    jobs = [(kind, spec.get("K"), spec.get("sigma"), spec.get("a"), arm, s)
            for arm in ("no_mine", "supcon") for s in range(N_SEEDS)]
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(_arm_job, jobs))
    nm = np.array([r["probe_acc"] for r in rows if r["arm"] == "no_mine"])
    sc = np.array([r["probe_acc"] for r in rows if r["arm"] == "supcon"])
    H = float(sc.mean() - nm.mean())
    floor = float(nm.std(ddof=1))
    hbin = next(b for b, lo, hi in H_BINS if lo <= max(H, 0) < hi)

    gam = None
    if gamma_ok and kind == "toy":
        rng = np.random.default_rng(3)
        x, _ = draw(rung, 1500, rng)
        _, gam, _ = gamma_probe(teacher_embed(rung, x), true_K=rung.K)

    a_ok = H > 2 * floor
    b_ok = S > 0.10
    c_ok = (rt_target is None) or (abs(rt - rt_target) <= 0.05)
    return dict(tag=rung.tag(), kind=kind, spec=spec, rho_tail=rt,
                rho_target=rt_target, S=S, H=H, noise_floor=floor,
                H_bin=hbin, gamma=gam,
                no_mine_accs=[float(v) for v in nm],
                supcon_accs=[float(v) for v in sc],
                calib_aborted=bool(calib is None or calib.aborted),
                lam=list(calib.lam) if calib and not calib.aborted else None,
                gate=dict(a=bool(a_ok), b=bool(b_ok), c=bool(c_ok)),
                full_arms=bool(a_ok and b_ok))


def main():
    t0 = time.time()
    cifar_patch()
    ks = json.load(open(os.path.join(RESULTS, "d2_knob_search.json")))
    gamma_ok = ks["gamma_validation"]["validated"]
    specs = []
    seen = set()
    for tgt, cands in ks["candidates"].items():
        for c in cands:
            key = (c["K"], c["sigma"], c["a"])
            if key in seen:
                continue
            seen.add(key)
            specs.append(dict(kind="toy", K=c["K"], sigma=c["sigma"],
                              a=c["a"], target=float(tgt)))
    specs.append(dict(kind="cifar"))

    rungs = []
    for spec in specs:
        r = preflight_rung(spec, gamma_ok)
        rungs.append(r)
        print(f"{r['tag']}: rho={r['rho_tail']:.3f} S={r['S']:.3f} "
              f"H={r['H']:.4f} floor={r['noise_floor']:.4f} bin={r['H_bin']} "
              f"gate={r['gate']} full={r['full_arms']}", flush=True)

    # designated rung: passing, closest to (H-high, rho 0.20)
    passing = [r for r in rungs if r["full_arms"] and r["kind"] == "toy"]
    desig = None
    if passing:
        desig = min(passing, key=lambda r: (abs(r["rho_tail"] - 0.20)
                                            - 0.001 * r["H"]))["tag"]
    out = dict(rungs=rungs, designated=desig, gamma_instrumented=gamma_ok,
               wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "d2_preflight.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("designated:", desig)


if __name__ == "__main__":
    main()
