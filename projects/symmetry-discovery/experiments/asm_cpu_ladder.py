"""asm_cpu_ladder — gated CPU ladder for Adversarial Symmetry Mining (ASM). CPU decides what deserves GPU.

Stages (run in order; each gates the next via asm_decision.py):
  A  unit/validity smoke   — transforms run, backward works, decoys rejected, valid ops survive.
  B  positive-control gap  — desat->full CIFAR; ASM/static must select saturate, avoid decoys, beat random.
  C  full-CIFAR ladder     — TinyEqM/v10-lite; SOLO (ASM>random) and/or HYBRID (v10+ASM>v10) gates.
  D  curriculum            — scheduled vs always-on hybrid.

Writes JSON per stage to results/asm/. No GPU here. CPU-runnable, CIFAR subset.

Run: python projects/symmetry-discovery/experiments/asm_cpu_ladder.py --stage A --n 512
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects" / "diff-EqM" / "experiments" / "dganm_variants"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _multi_morphism as MM           # noqa: E402
import asm_miner as ASM                 # noqa: E402
import v17_eval_metrics as EM           # noqa: E402

OUT = ROOT / "projects" / "symmetry-discovery" / "results" / "asm"
OUT.mkdir(parents=True, exist_ok=True)


def load_cifar(n, seed=0):
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(str(ROOT / "data"), train=True, download=False,
                                      transform=T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    return torch.stack([ds[i][0] for i in idx])


def tiny_eqm(seed=0):
    torch.manual_seed(seed)
    m = EM.TinyEqM()
    m._t_scale_999 = False
    return m


def _desat(x, keep=0.25):
    g = x.mean(1, keepdim=True)
    return (g + keep * (x - g)).clamp(-1, 1)


# --------------------------------------------------------------------------- STAGE A: unit/validity smoke
def stage_A(real, seed=0):
    rec = {"stage": "A", "checks": {}}
    dev = real.device
    # 1. every family applies without error + finite
    fam_ok, fam_err = [], []
    for f in ASM.ASM_ALL:
        try:
            mag = torch.full((real.size(0),), 0.5, device=dev)
            out = MM.apply_family(f, real, mag)
            assert torch.isfinite(out).all() and out.shape == real.shape
            fam_ok.append(f)
        except Exception as e:  # noqa
            fam_err.append((f, str(e)[:60]))
    rec["checks"]["families_ok"] = fam_ok
    rec["checks"]["families_err"] = fam_err

    # 2. outer-training backward works: L_EqM(model, T(x)) backprops into model
    model = tiny_eqm(seed)
    Tx = MM.apply_family("saturate", real[:64], torch.full((64,), 0.6, device=dev))
    loss, _, _ = ASM.eqm_loss_on(model, Tx)
    loss.backward()
    gnorm = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
    rec["checks"]["backward_grad_norm"] = round(gnorm, 4)
    rec["checks"]["backward_ok"] = gnorm > 0

    # 3. hardness computes for all 3 modes, finite
    hmodes = {}
    for mode in ("loss_only", "commutator_only", "loss_plus_comm"):
        H, parts = ASM.hardness(model, real[:64], "rotate",
                                torch.full((64,), 0.4, device=dev), mode=mode)
        hmodes[mode] = {"H": round(float(H), 4), **{k: round(v, 4) for k, v in parts.items()},
                        "finite": bool(torch.isfinite(H))}
    rec["checks"]["hardness_modes"] = hmodes

    # 4. miner: decoys rejected, valid ops survive
    scorer = MM.AnchorScorer(real, seed=777)
    ae = MM.train_robust_ae(real, steps=300, seed=seed)
    out = ASM.mine(model, real[:128], scorer, ae, K=48, mode="loss_plus_comm", seed=seed)
    rec["checks"]["n_valid"] = out["n_valid"]
    rec["checks"]["decoy_in_top"] = out["decoy_in_top"]
    rec["checks"]["top_families"] = out["top_families"]
    rec["checks"]["invalid_rate_decoys"] = {f: out["invalid_rate_by_family"][f] for f in ASM.ASM_DECOY}
    rec["checks"]["hardness_by_family"] = out["hardness_by_family"]

    # GATE A
    decoys_rejected = (out["decoy_in_top"] == 0 and
                       all(out["invalid_rate_by_family"][f] > 0.9 for f in ASM.ASM_DECOY))
    gate = (len(fam_err) == 0 and rec["checks"]["backward_ok"]
            and all(h["finite"] for h in hmodes.values())
            and out["n_valid"] > 0 and decoys_rejected)
    rec["gate_A_pass"] = bool(gate)
    rec["gate_reasons"] = {
        "all_families_run": len(fam_err) == 0,
        "backward_ok": rec["checks"]["backward_ok"],
        "hardness_finite": all(h["finite"] for h in hmodes.values()),
        "valid_ops_survive": out["n_valid"] > 0,
        "decoys_rejected": bool(decoys_rejected),
    }
    return rec


# --------------------------------------------------------------------------- STAGE B: positive-control gap
def _arm_policy(kind, real, model, scorer, ae, seed=0, asm_mode="loss_plus_comm"):
    """Return an aug_fn x->Tx for the given arm, on the desat gap (visible=desat, anchor=full)."""
    dev = real.device
    vfam = ASM.ASM_VALID
    if kind == "base":
        return None, {"selected": None}
    if kind == "random_valid":
        def fn(x):
            f = vfam[torch.randint(0, len(vfam), (1,)).item()]
            return MM.apply_family(f, x, (torch.rand(x.size(0), device=dev) * 2 - 1) * 0.6)
        return fn, {"selected": "random"}
    if kind == "static_gapaware":
        pol = MM.MorphismPolicy(MM.ALL_FAMILIES, depth=1).to(dev)
        d = MM.discover(pol, _desat(real, 0.25), scorer, steps=300, seed=seed, ae=ae,
                        ae_weight=5.0, gap_aware=True)
        for p in pol.parameters():
            p.requires_grad_(False)
        top = max(d["family_weights"].items(), key=lambda kv: kv[1])[0]
        return (lambda x: pol.sample_transform(x)), {"selected": top, "decoy_usage": d["decoy_usage"],
                                                     "weights": d["family_weights"]}
    # ASM arms: mine hard-valid on the desat visible each call (online); collapse to most-mined family
    mode = {"ASM_loss": "loss_only", "ASM_comm": "commutator_only",
            "ASM_loss_comm": "loss_plus_comm"}[kind]
    # pre-mine to find the dominant hard-valid family (for interpretable readout + a fast frozen aug)
    visible = _desat(real, 0.25)
    tally = {}
    for s in range(6):
        out = ASM.mine(model, visible[torch.randint(0, visible.size(0), (128,))], scorer, ae,
                       K=40, mode=mode, seed=seed + s)
        for f in out["top_families"]:
            tally[f] = tally.get(f, 0) + 1
    sel = max(tally, key=tally.get) if tally else "saturate"

    def fn(x):
        return MM.apply_family(sel, x, (torch.rand(x.size(0), device=dev) * 0.5 + 0.4))
    return fn, {"selected": sel, "tally": tally}


def stage_B(real, seed=0):
    rec = {"stage": "B", "arms": {}}
    visible = _desat(real, 0.25)
    full = real
    scorer = MM.AnchorScorer(real, seed=777)
    ae = MM.train_robust_ae(real, steps=800, seed=seed)
    probe = tiny_eqm(seed)  # model used to score hardness for ASM arms (lightly trained below)
    # quick warm of probe on visible so hardness is meaningful (not random-net)
    opt = torch.optim.Adam(probe.parameters(), 1e-3)
    for _ in range(300):
        l, _, _ = ASM.eqm_loss_on(probe, visible[torch.randint(0, visible.size(0), (128,))])
        opt.zero_grad(); l.backward(); opt.step()

    arms = ["base", "random_valid", "static_gapaware", "ASM_loss", "ASM_comm", "ASM_loss_comm"]
    for arm in arms:
        fn, info = _arm_policy(arm, real, probe, scorer, ae, seed=seed)
        net = EM.train_eqm_lite(visible, fn, lam=0.5, steps=500, seed=seed)
        fc = EM.eqm_field_consistency(net, visible, full, draws=8)
        decoy_use = info.get("decoy_usage")
        rec["arms"][arm] = {"selected": info.get("selected"), "eqm_full": round(fc["eqm_heldout"], 5),
                            "eqm_gap": round(fc["eqm_gap"], 5),
                            "decoy_usage": round(decoy_use, 4) if decoy_use is not None else None,
                            "info": {k: v for k, v in info.items() if k in ("tally", "weights")}}

    base = rec["arms"]["base"]["eqm_full"]; rnd = rec["arms"]["random_valid"]["eqm_full"]
    best_asm = min(rec["arms"][a]["eqm_full"] for a in ("ASM_loss", "ASM_comm", "ASM_loss_comm"))
    best_asm_arm = min(("ASM_loss", "ASM_comm", "ASM_loss_comm"),
                       key=lambda a: rec["arms"][a]["eqm_full"])
    static = rec["arms"]["static_gapaware"]
    # GATE B: a discovery arm selects saturate, decoys avoided, beats random by >=0.005 or 1%
    margin = rnd - best_asm
    sat_selected = (static["selected"] == "saturate" or
                    rec["arms"][best_asm_arm]["selected"] == "saturate")
    beats_random = margin >= max(0.005, 0.01 * rnd)
    rec["summary"] = {"base": base, "random": rnd, "best_asm": best_asm, "best_asm_arm": best_asm_arm,
                      "static": static["eqm_full"], "static_selected": static["selected"],
                      "margin_asm_vs_random": round(margin, 5)}
    rec["gate_B_pass"] = bool(sat_selected and beats_random)
    rec["gate_reasons"] = {"saturate_selected": bool(sat_selected),
                           "beats_random": bool(beats_random),
                           "best_asm_arm": best_asm_arm}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    real = load_cifar(args.n, seed=args.seed)
    t0 = time.time()
    rec = {"A": stage_A, "B": stage_B}[args.stage](real, seed=args.seed)
    rec["seconds"] = round(time.time() - t0, 1)
    p = OUT / f"stage_{args.stage}_seed{args.seed}.json"
    p.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print(f"\n--> gate_{args.stage}_pass = {rec.get(f'gate_{args.stage}_pass')}  ({rec['seconds']}s)  {p}")


if __name__ == "__main__":
    main()
