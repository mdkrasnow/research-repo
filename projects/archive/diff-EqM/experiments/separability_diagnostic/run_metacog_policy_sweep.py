"""GPU engine for the metacognition policy sweep (EqM B/2, inference-time).

Reuses the proven pareto_sample stack (model/VAE/Inception + incremental FID;
NEVER dumps 2048-d feats). Two engines:

  --engine selection  : run R full draws (--steps each) + read partial-traj
                        features; keep argmin policy.score. NFE/img = R*steps
                        EXACT — identical seeding to pareto_sample.py so the
                        energy_path arm reproduces pareto seed0 (repro check).
  --engine segmented  : run R lanes in segments; at read steps apply per-lane
                        actions (continue/restart/churn/eta/heun). Engine counts
                        EVERY model() forward -> exact measured NFE/img, written
                        to meta.json (aggregator flags off-budget arms).

Hard guarantees:
  * identical base z/y per slot across all arms (seed = f(slot,draw), paired).
  * no image pixels / no test-set quality labels used in any policy.
  * incremental FID stats only (s1=Σf, s2=Σfᵀf, n) per rank shard.

Outputs: <out>/stats_rank{r}.npz + <out>/meta_rank{r}.json (nfe, wall, policy cfg).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
VAE_SCALE = 0.18215


def _deps():
    here = Path(__file__).resolve().parent
    up = str(here.parent.parent / "eqm-upstream")
    e3 = str(here.parent / "exp3_fidelity_diversity")
    for p in (str(here), up, e3):
        if p not in sys.path:
            sys.path.insert(0, p)
    from models import EqM_models
    from download import find_model
    from diffusers.models import AutoencoderKL
    from features import _build_inception
    return EqM_models, find_model, AutoencoderKL, _build_inception


def _f(model, xt, t, y):
    out = model(xt, t, y)
    if not torch.is_tensor(out):
        out = out[0]
    return out.detach()


def gd_log(model, z, y, eta, steps):
    """Full descent; returns (xt, norm[B,steps-1], dot[B,steps-1]). 1 eval/step."""
    xt = z; t = torch.zeros((xt.shape[0],), device=xt.device)
    norms, dots = [], []
    for _ in range(steps - 1):
        out = _f(model, xt, t, y); xk = xt
        norms.append(out.flatten(1).norm(dim=1)); dots.append((out * xk).flatten(1).sum(1))
        xt = xk + out * eta
    return xt.detach(), torch.stack(norms, 1).float().cpu().numpy(), torch.stack(dots, 1).float().cpu().numpy()


def _seed_draw(so, slot, R, r):
    return so + int(slot) * R + r


def _z(dev, ls, seed, n=1):
    g = torch.Generator(device=dev).manual_seed(int(seed))
    return torch.randn(n, 4, ls, ls, generator=g, device=dev)


# --------------------------------------------------------------------------- #
def run_selection(args, model, vae, incep, dev, ls, rank, world):
    import metacog_policies as MP
    pol = MP.make_selection(args.policy, **json.loads(args.policy_kw))
    reads_k = sorted(set(pol.reads)) if pol.reads else []
    # load probe + stacked artifacts for every k any policy might need
    base = Path(args.probe_artifact).parent / "results" / "partial_probe"
    probe_art, stacked_art = {}, {}
    for k in set(reads_k) | {getattr(pol, "k", 50)}:
        pf = base / f"partial_probe_k{k}.npz"
        if pf.exists():
            d = np.load(pf, allow_pickle=True); probe_art[k] = {kk: d[kk] for kk in ["w", "b", "mu", "sd"]}
        sf = base / f"stacked_artifact_k{k}.npz"
        if sf.exists():
            d = np.load(sf, allow_pickle=True); stacked_art[k] = {kk: d[kk] for kk in ["w", "b", "mu", "sd"]}

    def feat(latents):
        im = vae.decode(latents / VAE_SCALE).sample
        im = (im / 2 + 0.5).clamp(0, 1)
        im = torch.nn.functional.interpolate(im, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
        f = incep(im)[0]
        if f.dim() == 4:
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)
        return f.float().cpu().numpy()

    s1 = np.zeros(2048); s2 = np.zeros((2048, 2048)); n = 0
    slots = list(range(rank, args.num_slots, world)); bs = args.batch_size
    so = args.seed_offset * (args.num_slots * 8 + 7); t0 = time.time()
    rng = np.random.default_rng(args.seed_offset * 7919 + rank)
    for j in range(0, len(slots), bs):
        chunk = slots[j:j + bs]; B = len(chunk); bi = np.arange(B)
        y = torch.tensor([(int(i) * 1315423911) % args.num_classes for i in chunk], device=dev)
        fr = np.zeros((args.R, B, 2048), np.float32)
        normR = [None] * args.R; dotR = [None] * args.R
        for r in range(args.R):
            z = torch.cat([_z(dev, ls, _seed_draw(so, i, args.R, r)) for i in chunk], 0)
            with torch.no_grad():
                if reads_k:
                    xt, nm, dt = gd_log(model, z, y, args.stepsize, args.steps)
                    normR[r] = nm; dotR[r] = dt
                else:
                    xt, _, _ = gd_log(model, z, y, args.stepsize, args.steps)
                fr[r] = feat(xt)
        reads = {}
        for k in reads_k:
            kk = min(k, normR[0].shape[1])
            reads[k] = {"norm": np.stack([normR[r][:, :kk] for r in range(args.R)]),
                        "dot": np.stack([dotR[r][:, :kk] for r in range(args.R)])}
        ctx = {"R": args.R, "B": B, "rng": rng, "probe_art": probe_art, "stacked_art": stacked_art}
        sc = pol.score(reads, ctx)            # (R,B) lower=keep
        pick = np.argmin(sc, 0)
        f = fr[pick, bi]
        s1 += f.sum(0); s2 += f.T @ f; n += B
        if rank == 0 and (j // bs) % 5 == 0:
            print(f"[sel:{args.policy}] {j+B}/{len(slots)} {(j+B)/max(1e-9,time.time()-t0):.1f}/s", flush=True)
    nfe_img = float(args.R * args.steps)
    return s1, s2, n, nfe_img, t0


# --------------------------------------------------------------------------- #
def run_segmented(args, model, vae, incep, dev, ls, rank, world):
    import metacog_policies as MP
    pol = MP.make_segmented(args.policy, **json.loads(args.policy_kw))
    reads_k = sorted({k for k in pol.reads if 0 < k < args.steps})  # reads must precede the end
    base = Path(args.probe_artifact).parent / "results" / "partial_probe"
    probe_art = {}
    for k in reads_k:
        d = np.load(base / f"partial_probe_k{k}.npz", allow_pickle=True)
        probe_art[k] = {kk: d[kk] for kk in ["w", "b", "mu", "sd"]}

    def feat(latents):
        im = vae.decode(latents / VAE_SCALE).sample
        im = (im / 2 + 0.5).clamp(0, 1)
        im = torch.nn.functional.interpolate(im, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
        f = incep(im)[0]
        if f.dim() == 4:
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)
        return f.float().cpu().numpy()

    s1 = np.zeros(2048); s2 = np.zeros((2048, 2048)); n = 0
    slots = list(range(rank, args.num_slots, world)); bs = args.batch_size
    so = args.seed_offset * (args.num_slots * 8 + 7); t0 = time.time()
    nfe_acc = 0.0; nfe_imgs = 0
    bounds = reads_k + [args.steps]
    for j in range(0, len(slots), bs):
        chunk = slots[j:j + bs]; B = len(chunk); bi = np.arange(B)
        y = torch.tensor([(int(i) * 1315423911) % args.num_classes for i in chunk], device=dev)
        t = torch.zeros((B,), device=dev)
        eta = np.full((args.R, B), args.stepsize)            # per-lane eta
        step_cnt = np.zeros((args.R, B))                     # exact NFE per (lane,img)
        xt = [None] * args.R; nbuf = [[] for _ in range(args.R)]; dbuf = [[] for _ in range(args.R)]
        restart_idx = 0
        for r in range(args.R):
            xt[r] = torch.cat([_z(dev, ls, _seed_draw(so, i, args.R, r)) for i in chunk], 0)
        prev = 0
        for seg_i, kb in enumerate(bounds):
            # advance every lane from prev -> kb steps
            for r in range(args.R):
                et = torch.tensor(eta[r], device=dev, dtype=torch.float32).view(B, 1, 1, 1)
                for _ in range(prev, kb - 1 if kb == args.steps else kb):
                    out = _f(model, xt[r], t, y)
                    nbuf[r].append(out.flatten(1).norm(dim=1).float().cpu().numpy())
                    dbuf[r].append((out * xt[r]).flatten(1).sum(1).float().cpu().numpy())
                    xt[r] = xt[r] + out * et
                    step_cnt[r] += 1.0
            prev = kb if kb != args.steps else (args.steps - 1)
            if kb in reads_k:                                # decision point
                nm = np.stack([np.stack(nbuf[r], 1) for r in range(args.R)])   # (R,B,kb)
                dt = np.stack([np.stack(dbuf[r], 1) for r in range(args.R)])
                R_, B_, kk = nm.shape
                risk = MP.probe_risk(nm.reshape(R_ * B_, kk), dt.reshape(R_ * B_, kk),
                                     probe_art[kb]).reshape(R_, B_)
                feats = MP.scalar_features(nm.reshape(R_ * B_, kk), dt.reshape(R_ * B_, kk))
                feats = {fk: fv.reshape(R_, B_) for fk, fv in feats.items()}
                a = pol.act(kb, risk, feats, {"R": args.R, "B": B}); act = a["action"]; prm = a["param"]
                for r in range(args.R):
                    m_restart = act[r] == 1; m_churn = act[r] == 2; m_eta = act[r] == 3
                    if m_restart.any():
                        new = torch.cat([_z(dev, ls, _seed_draw(so, i, args.R, args.R + restart_idx))
                                         for i in chunk], 0)
                        restart_idx += 1
                        mask = torch.tensor(m_restart, device=dev).view(B, 1, 1, 1)
                        xt[r] = torch.where(mask, new, xt[r])
                    if m_churn.any():
                        gk = torch.randn_like(xt[r]) * float(prm)
                        mask = torch.tensor(m_churn, device=dev).view(B, 1, 1, 1)
                        xt[r] = torch.where(mask, xt[r] + gk, xt[r])
                    if m_eta.any():
                        eta[r] = np.where(m_eta, eta[r] * float(prm), eta[r])
                # heun (action 4): one extra corrector eval on flagged lanes' next step
                # (counted below); applied as a single refined step here.
                for r in range(args.R):
                    m_heun = act[r] == 4
                    if m_heun.any():
                        et = torch.tensor(eta[r], device=dev, dtype=torch.float32).view(B, 1, 1, 1)
                        g1 = _f(model, xt[r], t, y)
                        xtmp = xt[r] + g1 * et
                        g2 = _f(model, xtmp, t, y)
                        upd = (g1 + g2) * 0.5 * et
                        mask = torch.tensor(m_heun, device=dev).view(B, 1, 1, 1)
                        # apply Heun update on flagged columns only; +1 extra eval there
                        xt[r] = torch.where(mask, xt[r] + upd, xt[r])
                        step_cnt[r] += np.where(m_heun, 1.0, 0.0)
        # final keep: argmin probe risk over full trajectory per lane
        nm = np.stack([np.stack(nbuf[r], 1) for r in range(args.R)])
        dt = np.stack([np.stack(dbuf[r], 1) for r in range(args.R)])
        kk = nm.shape[2]; kf = min(kk, max(reads_k))
        risk = MP.probe_risk(nm[:, :, :kf].reshape(args.R * B, kf),
                             dt[:, :, :kf].reshape(args.R * B, kf), probe_art[max(reads_k)]).reshape(args.R, B)
        pick = np.argmin(risk, 0)
        fr = np.stack([feat(xt[r]) for r in range(args.R)])
        f = fr[pick, bi]
        s1 += f.sum(0); s2 += f.T @ f; n += B
        nfe_acc += float(step_cnt.sum(0).sum()); nfe_imgs += B
        if rank == 0 and (j // bs) % 2 == 0:
            print(f"[seg:{args.policy}] {j+B}/{len(slots)} nfe/img~{nfe_acc/max(1,nfe_imgs):.0f}", flush=True)
    nfe_img = nfe_acc / max(1, nfe_imgs)
    return s1, s2, n, nfe_img, t0


def main(args):
    EqM_models, find_model, AutoencoderKL, build_incep = _deps()
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    device = int(os.environ.get("LOCAL_RANK", "0")); torch.cuda.set_device(device); dev = f"cuda:{device}"
    ls = args.image_size // 8
    model = EqM_models[args.model](input_size=ls, num_classes=args.num_classes, uncond=True, ebm="none").to(dev)
    st = find_model(args.ckpt); model.load_state_dict(st["ema"] if "ema" in st else st.get("model", st)); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_grad_enabled(False)   # inference-only; segmented feat()/decode must not build a graph
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(dev).eval()
    incep = build_incep(dev)
    fn = run_selection if args.engine == "selection" else run_segmented
    s1, s2, n, nfe_img, t0 = fn(args, model, vae, incep, dev, ls, rank, world)
    out = Path(args.out); os.makedirs(out, exist_ok=True)
    np.savez(out / f"stats_rank{rank}.npz", s1=s1, s2=s2, n=np.int64(n))
    (out / f"meta_rank{rank}.json").write_text(json.dumps(
        {"policy": args.policy, "engine": args.engine, "nfe_per_img": nfe_img,
         "nfe_target": float(args.R * args.steps), "n": int(n), "R": args.R, "steps": args.steps,
         "wall_s": round(time.time() - t0, 1), "policy_kw": args.policy_kw,
         "seed_offset": args.seed_offset}))
    print(f"[done] rank{rank} policy={args.policy} n={n} nfe/img={nfe_img:.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EqM-B/2"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probe-artifact", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=["selection", "segmented"], default="selection")
    ap.add_argument("--policy", required=True); ap.add_argument("--policy-kw", default="{}")
    ap.add_argument("--R", type=int, default=3); ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--num-slots", type=int, default=20000); ap.add_argument("--stepsize", type=float, default=0.003)
    ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-classes", type=int, default=1000); ap.add_argument("--vae", default="ema")
    ap.add_argument("--seed-offset", type=int, default=0)
    main(ap.parse_args())
