"""Phase 2 — online equal-NFE adaptive sampler (metacognition sampler).

The true metacognitive sampler: run GD, read a PARTIAL-trajectory risk score at a
decision step k_dec < N, and reallocate compute toward the flagged (likely-garbage)
slots — restart them — while spending the IDENTICAL extra compute randomly in the
control. partial_probe.py already showed the early risk score is real (de-conf
AUROC 0.814 at k=100/249). This script acts on it.

Arms (a fixed flag fraction f; expected NFE is IDENTICAL across the two restart arms):
  vanilla        : N steps, no adaptation.                    (un-adapted floor)
  random-restart : flag a RANDOM f of slots; each flagged slot draws ONE restart;
                   keep a random one of {orig, restart}.      (NEG: compute-matched)
  probe-restart  : flag top-f by partial risk; each flagged slot draws ONE restart;
                   keep the LOWER-risk of {orig, restart}.    (TREATMENT)
  oracle-restart : flag top-f by TRUE badness; keep the truly-better.  (POS ceiling)

NFE accounting (one GD step on one slot = 1 NFE):
  vanilla        = N
  random/probe   = N + f*N         (every slot runs N; flagged f fraction restart +N)
  oracle         = N + f*N         (same budget; cheats only on WHICH to flag/keep)
random-restart and probe-restart have EXACTLY equal expected NFE -> any gap is the
probe, not the compute. (vanilla is the cheaper un-adapted reference, labeled.)

Two execution modes:
  --mock : CPU. Synthetic trajectories with a known garbage latent; trains an inline
           partial probe; verifies NFE bookkeeping + that probe>random on synthetic
           quality. No EqM, no GPU. For logic/CI.
  (real) : GPU. Mirrors probe_gated_sample.py (lazy EqM deps, Inception features);
           restart = fresh noise, same class; quality = Inception NN-dist to real.
           Writes per-arm feature shards; FID via fid_gated_agg.py. Cluster-gated.

Run (smoke): python online_adaptive_sampler.py --mock --slots 4000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ----------------------------- shared probe ------------------------------- #
def load_partial_probe(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in ["w", "b", "mu", "sd"]} | {"k_dec": int(d["k_dec"])}


def probe_risk(norm_k, dot_k, art):
    """P(garbage) from a TRUNCATED [0:k] trajectory, using the saved feature spec."""
    from probe_validate import feature_groups
    X = feature_groups(norm_k, dot_k)["ALL-shape"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    z = ((X - art["mu"]) / art["sd"]) @ art["w"] + float(art["b"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def nfe_table(N, f):
    return {"vanilla": N, "random-restart": N * (1 + f),
            "probe-restart": N * (1 + f), "oracle-restart": N * (1 + f)}


# ----------------------------- mock mode ---------------------------------- #
def _synth(rng, n, T):
    """Garbage trajectories oscillate (zero-mean energy), good ones decay clean.
    Returns norm,dot (n,T), true garbage label, and a 'quality' (lower=better)."""
    g = (rng.uniform(0, 1, n) < 0.4).astype(float)            # 40% garbage
    base = np.linspace(40, 6, T)[None].repeat(n, 0)
    osc = np.zeros((n, T))
    for i in range(n):
        if g[i]:
            s, sgn = 0.0, 1.0
            for t in range(T):
                if rng.uniform() < 0.25:
                    sgn = -sgn
                s = 0.7 * s + 0.3 * sgn
                osc[i, t] = s
    norm = base + 3.0 * osc + rng.normal(0, 0.5, (n, T))
    dot = -(base ** 2) * 0.01 + rng.normal(0, 1, (n, T)) + 2.0 * osc
    quality = g * (1.0 + rng.normal(0, 0.2, n)) + (1 - g) * rng.normal(0, 0.2, n)
    return norm, dot, g, quality


def run_mock(args):
    rng = np.random.default_rng(args.seed)
    N, k, f = args.steps, int(args.kfrac * args.steps), args.flag_frac
    M = args.slots
    # train an inline partial probe on a held-out generation set
    n_tr = 1500
    ntr_norm, ntr_dot, ntr_g, _ = _synth(rng, n_tr, N)
    from probe_validate import feature_groups
    from learned_probe import fit_logreg, auc
    Xtr = np.nan_to_num(feature_groups(ntr_norm[:, :k], ntr_dot[:, :k])["ALL-shape"],
                        nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    w, b = fit_logreg((Xtr - mu) / sd, ntr_g)
    art = {"w": w, "b": np.float64(b), "mu": mu, "sd": sd, "k_dec": k}

    # generation pool: for each slot an original draw + ONE restart draw (cached)
    o_norm, o_dot, o_g, o_q = _synth(rng, M, N)
    r_norm, r_dot, r_g, r_q = _synth(rng, M, N)
    risk_o = probe_risk(o_norm[:, :k], o_dot[:, :k], art)
    risk_r = probe_risk(r_norm[:, :k], r_dot[:, :k], art)
    ho_auc = auc(o_g, risk_o)

    nflag = int(round(f * M))
    bi = np.arange(M)

    # vanilla: original draw only
    q_vanilla = o_q.copy()

    # probe-restart: flag top-f by orig risk; for flagged keep lower-risk{orig,restart}
    flag_p = np.zeros(M, bool); flag_p[np.argsort(-risk_o)[:nflag]] = True
    keep_restart_p = flag_p & (risk_r < risk_o)
    q_probe = np.where(keep_restart_p, r_q, o_q)

    # random-restart: flag random f; keep a random one of {orig,restart}
    flag_rnd = np.zeros(M, bool); flag_rnd[rng.permutation(M)[:nflag]] = True
    coin = rng.uniform(0, 1, M) < 0.5
    keep_restart_rnd = flag_rnd & coin
    q_random = np.where(keep_restart_rnd, r_q, o_q)

    # oracle-restart: flag top-f by TRUE badness; keep truly-better
    flag_or = np.zeros(M, bool); flag_or[np.argsort(-o_q)[:nflag]] = True
    keep_restart_or = flag_or & (r_q < o_q)
    q_oracle = np.where(keep_restart_or, r_q, o_q)

    # NFE check (count actual restarts)
    def nfe(flag):
        return N * M + N * int(flag.sum())
    nfe_p, nfe_rnd, nfe_or = nfe(flag_p), nfe(flag_rnd), nfe(flag_or)
    matched = nfe_p == nfe_rnd == nfe_or

    res = {"mode": "mock", "M": M, "N": N, "k_dec": k, "flag_frac": f,
           "partial_probe_auc": round(float(ho_auc), 4),
           "mean_quality_lower_is_better": {
               "vanilla": round(float(q_vanilla.mean()), 4),
               "random-restart": round(float(q_random.mean()), 4),
               "probe-restart": round(float(q_probe.mean()), 4),
               "oracle-restart": round(float(q_oracle.mean()), 4)},
           "probe_minus_random": round(float(q_probe.mean() - q_random.mean()), 4),
           "nfe": {"vanilla": N * M, "random-restart": nfe_rnd,
                   "probe-restart": nfe_p, "oracle-restart": nfe_or},
           "nfe_matched_probe_vs_random": bool(matched)}
    gap = q_random.mean() - q_probe.mean()   # positive => probe better (lower q)
    res["verdict"] = ("PROBE>RANDOM at equal NFE (logic OK)" if gap > 0.01 and matched
                      else "PROBE≈RANDOM" if matched else "NFE-MISMATCH-BUG")
    out = Path(args.out) if args.out else Path(__file__).parent / "results" / "online_adaptive"
    out.mkdir(parents=True, exist_ok=True)
    (out / "online_mock.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


# ----------------------------- real (GPU) mode ---------------------------- #
def _deps():
    up = str(Path(__file__).resolve().parents[2] / "eqm-upstream")
    e3 = str(Path(__file__).resolve().parent.parent / "exp3_fidelity_diversity")
    for p in (str(Path(__file__).resolve().parent), up, e3):
        if p not in sys.path:
            sys.path.insert(0, p)
    from models import EqM_models
    from download import find_model
    from diffusers.models import AutoencoderKL
    from features import _build_inception
    return EqM_models, find_model, AutoencoderKL, _build_inception


def gd_partial(model, z, y, eta, steps, k_dec):
    """Run GD; return (latent@k_dec, latent@N, norm[:k], dot[:k]) for online decisions.
    Logs only up to k_dec (the causal decision window)."""
    import torch
    xt = z
    t = torch.zeros((xt.shape[0],), device=xt.device)
    norms, dots = [], []
    x_at_k = None
    for s in range(steps - 1):
        out = model(xt, t, y)
        if not torch.is_tensor(out):
            out = out[0]
        out = out.detach(); xk = xt.detach()
        if s < k_dec:
            norms.append(out.flatten(1).norm(dim=1))
            dots.append((out * xk).flatten(1).sum(dim=1))
        xt = xk + out * eta
        if s == k_dec - 1:
            x_at_k = xt.detach().clone()
    import torch as _t
    norm = _t.stack(norms, 1).float().cpu().numpy()
    dot = _t.stack(dots, 1).float().cpu().numpy()
    return x_at_k, xt.detach(), norm, dot


def gd_continue(model, x_at_k, y, eta, steps, k_dec):
    """Finish a trajectory from the cached k_dec state to N."""
    import torch
    xt = x_at_k
    t = torch.zeros((xt.shape[0],), device=xt.device)
    for _ in range(steps - 1 - k_dec):
        out = model(xt, t, y)
        if not torch.is_tensor(out):
            out = out[0]
        xt = xt.detach() + out.detach() * eta
    return xt.detach()


def run_real(args):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    EqM_models, find_model, AutoencoderKL, build_incep = _deps()
    from features import inception_features
    VAE_SCALE = 0.18215
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device); dev = f"cuda:{device}"
    ls = args.image_size // 8
    N, f = args.num_sampling_steps, args.flag_frac
    art = load_partial_probe(args.partial_probe)
    k_dec = art["k_dec"] if args.k_dec < 0 else args.k_dec

    model = EqM_models[args.model](input_size=ls, num_classes=args.num_classes,
                                   uncond=True, ebm="none").to(dev).eval()
    st = find_model(args.ckpt)
    model.load_state_dict(st["ema"] if "ema" in st else st.get("model", st))
    for p in model.parameters():
        p.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(dev).eval()
    incep = build_incep(dev)

    bank_path = Path(args.out) / "real_bank.npy"
    if rank == 0 and not bank_path.exists():
        from compute_quality_labels import list_real_images
        rf, _ = inception_features("", device=dev, batch_size=args.batch_size,
                                   files=list_real_images(args.real_dir, args.num_real_bank, seed=0))
        os.makedirs(args.out, exist_ok=True); np.save(bank_path, rf.astype(np.float32))
    for _ in range(600):
        if bank_path.exists():
            break
        time.sleep(1)
    real_bank = torch.tensor(np.load(bank_path), device=dev)

    def incep_feat(lat):
        imgs = vae.decode(lat / VAE_SCALE).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = torch.nn.functional.interpolate(imgs, (299, 299), mode="bilinear",
                                               align_corners=False, antialias=True)
        ff = incep(imgs)[0]
        if ff.dim() == 4:
            ff = torch.nn.functional.adaptive_avg_pool2d(ff, 1).squeeze(-1).squeeze(-1)
        return ff

    slots = list(range(rank, args.num_slots, world)); bs = args.batch_size
    rng = np.random.default_rng(args.seed + rank)
    feats = {a: [] for a in ["vanilla", "random-restart", "probe-restart", "oracle-restart"]}
    t0 = time.time()
    for s0 in range(0, len(slots), bs):
        chunk = slots[s0:s0 + bs]; B = len(chunk)
        y = torch.tensor([(int(i) * 1315423911) % args.num_classes for i in chunk], device=dev)
        # original draw
        g0 = [torch.Generator(device=dev).manual_seed(int(i) * 2) for i in chunk]
        z0 = torch.stack([torch.randn(4, ls, ls, generator=g, device=dev) for g in g0])
        with torch.no_grad():
            xk0, xn0, nrm0, dot0 = gd_partial(model, z0, y, args.stepsize, N, k_dec)
            f0 = incep_feat(xn0); d0 = torch.cdist(f0, real_bank).topk(3, largest=False).values.mean(1)
        risk0 = probe_risk(nrm0, dot0, art)
        # restart draw (fresh noise, same class)
        g1 = [torch.Generator(device=dev).manual_seed(int(i) * 2 + 1) for i in chunk]
        z1 = torch.stack([torch.randn(4, ls, ls, generator=g, device=dev) for g in g1])
        with torch.no_grad():
            xk1, xn1, nrm1, dot1 = gd_partial(model, z1, y, args.stepsize, N, k_dec)
            f1 = incep_feat(xn1); d1 = torch.cdist(f1, real_bank).topk(3, largest=False).values.mean(1)
        risk1 = probe_risk(nrm1, dot1, art)
        f0n, f1n = f0.cpu().numpy(), f1.cpu().numpy()
        d0n, d1n = d0.cpu().numpy(), d1.cpu().numpy()
        nflag = int(round(f * B)); idx = np.arange(B)
        # arms
        feats["vanilla"].append(f0n)
        # probe: flag top risk, keep lower-risk of two
        fp = np.zeros(B, bool); fp[np.argsort(-risk0)[:nflag]] = True
        useR = fp & (risk1 < risk0)
        feats["probe-restart"].append(np.where(useR[:, None], f1n, f0n))
        # random: flag random, keep random of two
        fr = np.zeros(B, bool); fr[rng.permutation(B)[:nflag]] = True
        useRr = fr & (rng.uniform(0, 1, B) < 0.5)
        feats["random-restart"].append(np.where(useRr[:, None], f1n, f0n))
        # oracle: flag worst by true dist, keep truly-better
        fo = np.zeros(B, bool); fo[np.argsort(-d0n)[:nflag]] = True
        useRo = fo & (d1n < d0n)
        feats["oracle-restart"].append(np.where(useRo[:, None], f1n, f0n))
        if rank == 0 and (s0 // bs) % 5 == 0:
            print(f"[online] {s0+B}/{len(slots)} {(s0+B)/max(1e-9,time.time()-t0):.1f} slot/s", flush=True)
    out = Path(args.out); os.makedirs(out, exist_ok=True)
    for a in feats:
        arr = np.concatenate(feats[a], 0).astype(np.float32) if feats[a] else np.zeros((0, 2048), np.float32)
        np.save(out / f"feat_{a.replace('-', '_')}_rank{rank}.npy", arr)
    meta = {"N": N, "k_dec": k_dec, "flag_frac": f,
            "nfe": nfe_table(N, f), "num_slots": args.num_slots}
    (out / f"meta_rank{rank}.json").write_text(json.dumps(meta, indent=2))
    print(f"[online] rank{rank} DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--slots", type=int, default=4000)        # mock
    ap.add_argument("--steps", type=int, default=249)         # mock N
    ap.add_argument("--kfrac", type=float, default=0.4)       # mock decision point
    ap.add_argument("--flag-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    # real
    ap.add_argument("--model", default="EqM-B/2")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--partial-probe", default="")
    ap.add_argument("--k-dec", type=int, default=-1)
    ap.add_argument("--real-dir", default="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train")
    ap.add_argument("--num-real-bank", type=int, default=10000)
    ap.add_argument("--num-slots", type=int, default=15000)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--vae", default="ema")
    ap.add_argument("--stepsize", type=float, default=0.003)
    ap.add_argument("--num-sampling-steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    if args.mock:
        run_mock(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
