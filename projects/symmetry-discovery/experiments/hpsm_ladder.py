"""hpsm_ladder — gated CPU ladder for Hard-Positive Symmetry Mining (HPSM). CPU decides GPU.

A  unit/validity smoke  — transforms shape/range/NaN; miner returns valid T*; decoys penalized;
                          v18 + v19 step_fn forward/backward; lam=0 reductions.
B  positive-control gap — desat->full CIFAR; HPSM must select saturate (decoy<0.05) + beat random.
C  full-CIFAR           — TinyEqM v10-lite; SOLO (HPSM>random&base) + HYBRID (v10+HPSM>v10).

Writes results/hpsm/stage_*.json + summary.json + a per-stage decision (ADVANCE/REPAIR/STOP/PROMOTE_TO_GPU).
No GPU. CPU-runnable.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects" / "diff-EqM" / "experiments" / "dganm_variants"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _multi_morphism as MM    # noqa: E402
import asm_miner as ASM         # noqa: E402
import hpsm_miner as HP         # noqa: E402
import v17_eval_metrics as EM   # noqa: E402

OUT = ROOT / "projects" / "symmetry-discovery" / "results" / "hpsm"
OUT.mkdir(parents=True, exist_ok=True)


def load_cifar(n, seed=0):
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(str(ROOT / "data"), train=True, download=False,
                                      transform=T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    return torch.stack([ds[i][0] for i in idx])


def _desat(x, keep=0.25):
    g = x.mean(1, keepdim=True)
    return (g + keep * (x - g)).clamp(-1, 1)


class _TinyUNet(nn.Module):
    """Stand-in velocity field model(x, t)->velocity for fwd/bwd smoke (t in [0,999], ignored)."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1); self.c2 = nn.Conv2d(32, 32, 3, 1, 1); self.c3 = nn.Conv2d(32, 3, 3, 1, 1)
        self._t_scale_999 = True

    def forward(self, x, t):
        h = F.relu(self.c1(x)); h = F.relu(self.c2(h)); return self.c3(h)


# --------------------------------------------------------------------------- STAGE A
def stage_A(real, seed=0):
    rec = {"stage": "A", "checks": {}}
    dev = real.device

    fam_err = []
    for f in HP.ALL:
        try:
            out = MM.apply_family(f, real[:32], torch.full((32,), 0.5, device=dev))
            assert out.shape == real[:32].shape and torch.isfinite(out).all()
            assert out.min() >= -1.01 and out.max() <= 1.01
        except Exception as ex:  # noqa
            fam_err.append((f, str(ex)[:50]))
    rec["checks"]["families_ok"] = len(HP.ALL) - len(fam_err)
    rec["checks"]["families_err"] = fam_err

    scorer = MM.AnchorScorer(real, seed=777)
    ae = MM.train_robust_ae(real, steps=300, seed=seed)
    model = _TinyUNet().to(dev)
    # miner returns valid T*; high decoy penalty -> decoy_usage < 0.05
    T_star, diag = HP.mine(model, _desat(real[:128]), scorer, ae, K=32, mode="loss_plus_comm",
                           w_decoy=50.0, seed=seed)
    rec["checks"]["T_star"] = diag["T_star"]
    rec["checks"]["decoy_usage_highpenalty"] = diag["decoy_usage"]
    rec["checks"]["n_valid"] = diag["n_valid"]
    rec["checks"]["validity_by_family_decoys"] = {f: diag["validity_by_family"][f] for f in HP.DECOY}

    # v18 / v19 forward+backward via step_fn (set module state manually)
    import v18_hpsm_morph as V18
    import v19_anm_hpsm_hybrid as V19
    from _common import TrainArgs
    args = TrainArgs(variant="v18", output_dir="/tmp/x", train_eps=1e-3, a=0.8, gain=4.0)
    x = real[:16]

    def run_v18(lam_hpsm):
        V18._H.clear()
        V18._H.update({"lam_hpsm": lam_hpsm, "lam_consistency": 0.1, "hpsm_k": 6,
                       "hpsm_mode": "loss_plus_comm", "warmup_epochs": 0, "steps_per_epoch": 1,
                       "w_gap": 2.0, "decoy_weight": 10.0, "mine_every": 1,
                       "scorer": scorer if lam_hpsm > 0 else None, "ae": ae})
        m = _TinyUNet().to(dev)
        tot, d = V18.step_fn(m, x, 5, dev, args)
        tot.backward()
        gn = sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
        return float(tot), d, gn

    t18, d18, g18 = run_v18(0.3)
    t18b, d18b, _ = run_v18(0.0)  # lam_hpsm=0 -> base
    rec["checks"]["v18"] = {"total": round(t18, 4), "base": round(d18["base"], 4),
                            "hpsm": round(d18["hpsm"], 4), "cons": round(d18["cons"], 4),
                            "grad_norm": round(g18, 4), "finite": bool(torch.isfinite(torch.tensor(t18))),
                            "lam0_reduces_base": abs(d18b["hpsm"]) < 1e-9}

    def run_v19(lam_anm, lam_hpsm):
        V19._H.clear()
        V19._H.update({"lam_anm": lam_anm, "lam_hpsm": lam_hpsm, "lam_consistency": 0.1,
                       "eps_radius": 0.3, "mining_lr": 0.05, "mining_K": 1, "hpsm_k": 6,
                       "hpsm_mode": "loss_plus_comm", "warmup_epochs": 0, "w_gap": 2.0,
                       "decoy_weight": 10.0, "steps_per_epoch": 1,
                       "scorer": scorer if lam_hpsm > 0 else None, "ae": ae})
        m = _TinyUNet().to(dev)
        tot, d = V19.step_fn(m, x, 5, dev, args)
        tot.backward()
        gn = sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
        return float(tot), d, gn

    t19, d19, g19 = run_v19(0.1, 0.3)
    _, d19a, _ = run_v19(0.0, 0.3)  # anm=0 -> hpsm only
    rec["checks"]["v19"] = {"total": round(t19, 4), "base": round(d19["base"], 4),
                            "anm": round(d19["anm"], 4), "hpsm": round(d19["hpsm"], 4),
                            "cons": round(d19["cons"], 4), "grad_norm": round(g19, 4),
                            "all_components_nonzero": all(abs(d19[k]) > 1e-9 for k in ("base", "anm", "hpsm")),
                            "anm0_reduces_hpsm": abs(d19a["anm"]) < 1e-9}

    decoys_penalized = (diag["decoy_usage"] < 0.05 and diag["T_star"]["family"] not in HP.DECOY)
    gate = (len(fam_err) == 0 and decoys_penalized and rec["checks"]["v18"]["finite"]
            and g18 > 0 and g19 > 0 and rec["checks"]["v19"]["all_components_nonzero"]
            and rec["checks"]["v18"]["lam0_reduces_base"] and rec["checks"]["v19"]["anm0_reduces_hpsm"])
    rec["gate_A_pass"] = bool(gate)
    rec["decision"] = "ADVANCE" if gate else "REPAIR"
    return rec


# --------------------------------------------------------------------------- STAGE B / C share a probe trainer
def _warm_probe(train, steps=120, seed=0):
    m = EM.TinyEqM(); m._t_scale_999 = False
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    n = train.size(0)
    for _ in range(steps):
        l, _, _ = ASM.eqm_loss_on(m, train[torch.randint(0, n, (128,))]); opt.zero_grad(); l.backward(); opt.step()
    return m


def _hpsm_family(model, visible, scorer, ae, mode, seed=0, rounds=3):
    tally = {}
    for s in range(rounds):
        T_star, d = HP.mine(model, visible[torch.randint(0, visible.size(0), (96,))], scorer, ae,
                            K=24, mode=mode, w_gap=2.0, seed=seed + s)
        tally[T_star[0]] = tally.get(T_star[0], 0) + 1
    return max(tally, key=tally.get), tally


def _mine_delta_tiny(net, x_t, gamma, target, eps_radius=0.3, lr=0.05, K=1):
    """v10 ANM hard-negative PGA on TinyEqM (noise-space)."""
    d = torch.zeros_like(x_t).normal_(0, eps_radius / 2)
    n = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1); d = d * (eps_radius / (n + 1e-8)).clamp(max=1.0)
    for _ in range(K):
        d = d.detach().requires_grad_(True)
        loss = ((ASM.eqm_field(net, x_t + d, gamma) - target) ** 2).mean()
        g = torch.autograd.grad(loss, d)[0]
        with torch.no_grad():
            d = d + lr * g.sign()
            n = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1); d = d * (eps_radius / (n + 1e-8)).clamp(max=1.0)
    return d.detach()


def train_eqm_arm(visible, kind, scorer, ae, spol=None, mode="loss_plus_comm", consistency=True,
                  mine_v10=False, lam_v10=0.1, steps=220, bs=96, lam=0.5, lam_c=0.3, mine_every=5, K=8, seed=0):
    """Train TinyEqM with the ACTUAL HPSM objective (the rewrite). kind:
      base   : no aug.
      random : ONLINE random valid (fam,mag) fresh each batch — matched aug strength to HPSM.
      static : the learned gap-aware policy (v17/gap15 approach).
      hpsm   : ONLINE re-mine the hardest VALID (fam,mag) against the CURRENT field every mine_every steps,
               apply the MINED magnitude, + lam_c*commutator (teach field to ACCEPT the symmetry).
    The earlier proxy froze HPSM to one family + random magnitude + no consistency -> not HPSM. This is."""
    torch.manual_seed(seed)
    net = EM.TinyEqM(); net._t_scale_999 = False
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    n = visible.size(0); dev = visible.device; vfam = HP.VALID
    fam_tally = {}
    cached = None  # (fam, mag) for hpsm between re-mines
    for step in range(steps):
        xb = visible[torch.randint(0, n, (bs,), device=dev)]
        base, _, _ = ASM.eqm_loss_on(net, xb)
        loss = base
        fam = None; magv = None
        if kind == "random":
            fam = vfam[int(torch.randint(0, len(vfam), (1,)))]
            magv = float((torch.rand(1) * 2 - 1) * 0.8)
        elif kind == "static":
            with torch.no_grad():
                Tx = spol.sample_transform(xb)
        elif kind == "hpsm":
            if cached is None or step % mine_every == 0:
                with torch.no_grad():
                    cached, _ = HP.mine(net, xb, scorer, ae, K=K, mode=mode, w_gap=2.0,
                                        w_decoy=20.0, seed=seed * 1000 + step)
            fam, magv = cached
        if kind in ("random", "hpsm"):
            mag = torch.full((bs,), float(magv), device=dev)
            Tx = MM.apply_family(fam, xb, mag)
            fam_tally[fam] = fam_tally.get(fam, 0) + 1
        if kind != "base":
            aug, _, _ = ASM.eqm_loss_on(net, Tx.detach())
            loss = loss + lam * aug
            if kind == "hpsm" and consistency:
                loss = loss + lam_c * HP.commutator_consistency(net, xb, fam, mag)
        if mine_v10:  # v10 ANM hard-negative term (for v10_lite + hybrid arms)
            gamma = torch.rand(bs, device=dev) * 0.998 + 0.001
            eps = torch.randn_like(xb); g = gamma.view(-1, 1, 1, 1)
            x_t = (1 - g) * eps + g * xb; tgt = eps - xb
            delta = _mine_delta_tiny(net, x_t, gamma, tgt)
            lh = ((ASM.eqm_field(net, x_t + delta, gamma) - tgt) ** 2).mean()
            loss = loss + lam_v10 * lh
        opt.zero_grad(); loss.backward(); opt.step()
    sel = max(fam_tally, key=fam_tally.get) if fam_tally else (kind if kind != "hpsm" else "saturate")
    return net, sel, fam_tally


def stage_B(real, seed=0):
    rec = {"stage": "B", "arms": {}}
    visible = _desat(real, 0.25); full = real
    scorer = MM.AnchorScorer(real, seed=777); ae = MM.train_robust_ae(real, steps=300, seed=seed)
    # static gap-aware policy (the gap15 approach) — discovered once, used as a fixed aug policy.
    spol = MM.MorphismPolicy(MM.ALL_FAMILIES, depth=1).to(real.device)
    sd = MM.discover(spol, visible, scorer, steps=150, seed=seed, ae=ae, ae_weight=5.0, gap_aware=True)
    for p in spol.parameters():
        p.requires_grad_(False)
    static_sel = max(sd["family_weights"].items(), key=lambda kv: kv[1])[0]

    arms = {
        "base": dict(kind="base"),
        "random_valid": dict(kind="random"),
        "static_gapaware": dict(kind="static", spol=spol),
        "HPSM": dict(kind="hpsm", mode="loss_plus_comm", consistency=True),
        "HPSM_noconsist": dict(kind="hpsm", mode="loss_plus_comm", consistency=False),  # ablation
        "HPSM_lossonly": dict(kind="hpsm", mode="loss_only", consistency=True),         # ablation
    }
    for arm, kw in arms.items():
        net, sel, tally = train_eqm_arm(visible, scorer=scorer, ae=ae, seed=seed, **kw)
        fc = EM.eqm_field_consistency(net, visible, full, draws=6)
        decoy = sum(tally.get(f, 0) for f in HP.DECOY) / max(sum(tally.values()), 1) if tally else 0.0
        rec["arms"][arm] = {"eqm_full": round(fc["eqm_heldout"], 5),
                            "selected": static_sel if arm == "static_gapaware" else sel,
                            "decoy_usage": round(decoy, 4), "tally": tally}

    rnd = rec["arms"]["random_valid"]["eqm_full"]
    hpsm = rec["arms"]["HPSM"]["eqm_full"]
    hpsm_sel = rec["arms"]["HPSM"]["selected"]
    hpsm_decoy = rec["arms"]["HPSM"]["decoy_usage"]
    sat = (hpsm_sel == "saturate" or static_sel == "saturate")
    beats = (rnd - hpsm) >= max(0.005, 0.01 * rnd)
    rec["summary"] = {"base": rec["arms"]["base"]["eqm_full"], "random": rnd, "static": rec["arms"]["static_gapaware"]["eqm_full"],
                      "HPSM": hpsm, "HPSM_selected": hpsm_sel, "HPSM_decoy": hpsm_decoy,
                      "margin_hpsm_vs_random": round(rnd - hpsm, 5),
                      "ablation_noconsist": rec["arms"]["HPSM_noconsist"]["eqm_full"],
                      "ablation_lossonly": rec["arms"]["HPSM_lossonly"]["eqm_full"]}
    rec["gate_B_pass"] = bool(sat and beats and hpsm_decoy < 0.05)
    rec["decision"] = "ADVANCE" if rec["gate_B_pass"] else ("REPAIR" if not sat or hpsm_decoy >= 0.05 else "STOP")
    return rec


def stage_C(real, seed=0):
    """Full CIFAR, NO gap. Tests whether the consistency (equivariance) mechanism that won Stage B also
    helps WITHOUT a missing-factor gap. Same online trainer; visible=full train split, eval=full held."""
    rec = {"stage": "C", "arms": {}}
    ntr = min(real.size(0) - 128, 384)
    train, held = real[:ntr], real[ntr:ntr + 128]
    scorer = MM.AnchorScorer(real, seed=777); ae = MM.train_robust_ae(real, steps=300, seed=seed)
    # static v17 full-CIFAR discovery (gap_aware=False, center-zoom visible)
    cc = int(round(32 / 1.5)); off = (32 - cc) // 2
    vis = F.interpolate(real[:, :, off:off + cc, off:off + cc], size=32, mode="bilinear", align_corners=False)
    spol = MM.MorphismPolicy(MM.ALL_FAMILIES, depth=1).to(real.device)
    MM.discover(spol, vis, scorer, steps=120, seed=seed, ae=ae, ae_weight=50.0, gap_aware=False)
    for p in spol.parameters():
        p.requires_grad_(False)

    arms = {
        "base": dict(kind="base"),
        "random_valid": dict(kind="random"),
        "static_v17": dict(kind="static", spol=spol),
        "HPSM": dict(kind="hpsm", mode="loss_plus_comm", consistency=True),
        "v10_lite": dict(kind="base", mine_v10=True),
        "v10+HPSM": dict(kind="hpsm", mode="loss_plus_comm", consistency=True, mine_v10=True),
    }
    for arm, kw in arms.items():
        net, sel, tally = train_eqm_arm(train, scorer=scorer, ae=ae, seed=seed, **kw)
        eqm = round(float(EM.eqm_loss(net, held, draws=6)), 5)
        rec["arms"][arm] = {"eqm_full": eqm, "selected": sel}

    base = rec["arms"]["base"]["eqm_full"]; rnd = rec["arms"]["random_valid"]["eqm_full"]
    hpsm = rec["arms"]["HPSM"]["eqm_full"]; v10 = rec["arms"]["v10_lite"]["eqm_full"]
    hyb = rec["arms"]["v10+HPSM"]["eqm_full"]
    solo = (hpsm < rnd) and (hpsm < base)
    hybrid = (v10 - hyb) >= max(0.005, 0.01 * v10)
    rec["summary"] = {"base": base, "random": rnd, "HPSM": hpsm, "v10_lite": v10, "hybrid": hyb,
                      "solo_margin_vs_random": round(rnd - hpsm, 5), "hybrid_margin_vs_v10": round(v10 - hyb, 5)}
    rec["gate_solo_pass"] = bool(solo); rec["gate_hybrid_pass"] = bool(hybrid)
    rec["gate_C_pass"] = bool(solo or hybrid)
    rec["decision"] = "PROMOTE_TO_GPU" if (solo or hybrid) else "STOP"
    return rec


def update_summary():
    s = {}
    for st in ("A", "B", "C"):
        p = OUT / f"stage_{st}.json"
        if p.exists():
            d = json.loads(p.read_text())
            s[st] = {"gate": d.get(f"gate_{st}_pass", d.get("gate_C_pass")), "decision": d.get("decision")}
    (OUT / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B", "C"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(8)
    real = load_cifar(args.n, seed=args.seed)
    t0 = time.time()
    rec = {"A": stage_A, "B": stage_B, "C": stage_C}[args.stage](real, seed=args.seed)
    rec["seconds"] = round(time.time() - t0, 1)
    (OUT / f"stage_{args.stage}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    summ = update_summary()
    print(f"\n--> stage {args.stage} decision = {rec.get('decision')}  ({rec['seconds']}s)")
    print(f"--> summary: {summ}")


if __name__ == "__main__":
    main()
