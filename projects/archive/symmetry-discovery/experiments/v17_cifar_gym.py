"""v17_cifar_gym — CIFAR natural-image morphism-discovery ladder with a REAL visible->anchor gap.

Why this exists: the prior CIFAR bridge was a WEAK NEGATIVE because visible and anchor were both
ordinary CIFAR, so any on-manifold-moving op looked equally good and decoys (shear) leaked. The v17
synthetic gym worked precisely because visible was a NARROWED view and anchor was the broader valid
distribution, so the gap PINNED which morphism family is needed. This script reconstructs that gap on
real CIFAR: each task builds VISIBLE by removing a specific factor; discovery must find the family that
re-introduces it (and reject decoys).

Pipeline: CONSTRUCT tasks -> CALIBRATE (visible!=anchor, per-family ED leakage, AE content proxy) ->
DISCOVER (only on calibrated tasks) -> short PAYOFF proxy (visible->anchor ED for BASE/RANDOM_VALID/
RANDOM_WITH_DECOYS/DISCOVERED/NO_ANCHOR/NO_DIVERSITY/ORACLE_VALID).

CPU-local, no FID, no pretrained semantic encoders. GT factors NOT used (gap is constructed by the
task, scored by the label-free anchor). Reuses morphisms/anchor/AE/policy/discover from the diff-EqM
bridge module `_multi_morphism`.

Run:  python projects/symmetry-discovery/experiments/v17_cifar_gym.py --stage all --seed 0
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects" / "diff-EqM" / "experiments" / "dganm_variants"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _multi_morphism as MM  # noqa: E402
import v17_eval_metrics as EM  # noqa: E402  (TinyEqM, train_eqm_lite, eqm_field_consistency)

OUT = ROOT / "projects" / "symmetry-discovery" / "results" / "cifar_gym"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------- saturation valid family (lowcolor)
def m_saturate(img, mag):
    gray = img.mean(1, keepdim=True)
    k = 1.0 + mag.view(-1, 1, 1, 1)  # mag in [-rng,rng]; >0 boosts chroma, <0 desaturates
    return (gray + k * (img - gray)).clamp(-1, 1)


# register so MM.apply_family / decoy_usage see it (additive; bridge defaults untouched)
MM.VALID_FAMILIES["saturate"] = (m_saturate, 0.8)


# --------------------------------------------------------------------- data
def load_cifar(n, seed=0):
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(str(ROOT / "data"), train=True, download=False,
                                      transform=T.Compose([T.ToTensor(),
                                                           T.Normalize([0.5] * 3, [0.5] * 3)]))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    return torch.stack([ds[i][0] for i in idx])  # [-1,1]


# --------------------------------------------------------------------- task specs (construct visible)
def _centerzoom(real, zoom=1.6):
    cc = int(round(32 / zoom)); off = (32 - cc) // 2
    return F.interpolate(real[:, :, off:off + cc, off:off + cc], size=32,
                         mode="bilinear", align_corners=False)


def _lowcolor(real, keep=0.25):
    gray = real.mean(1, keepdim=True)
    return (gray + keep * (real - gray)).clamp(-1, 1)


def _position(real, shift=0.25):
    # push objects to a fixed corner (lose position diversity) -> recover with translate/scale
    th = torch.zeros(real.size(0), 2, 3, device=real.device)
    th[:, 0, 0] = 1; th[:, 1, 1] = 1; th[:, 0, 2] = shift; th[:, 1, 2] = shift
    return MM._affine(real, th)


def _shift_only(real, dx=0.25):
    # gap is a pure translation -> rotation should NOT help
    th = torch.zeros(real.size(0), 2, 3, device=real.device)
    th[:, 0, 0] = 1; th[:, 1, 1] = 1; th[:, 0, 2] = dx
    return MM._affine(real, th)


# task -> (visible_fn, expected_valid_families, note)
TASKS = {
    "centerzoom_to_full": (_centerzoom, {"scale", "translate_x", "translate_y"},
                           "objects enlarged+centered; recover scale-down + translate, not rotate/shear"),
    "lowcolor_to_full":   (_lowcolor, {"saturate", "hue", "bright"},
                           "desaturated; recover color/saturation/brightness, not spatial"),
    "position_to_full":   (_position, {"translate_x", "translate_y", "scale"},
                           "objects pushed to corner; recover translate/scale"),
    "no_rotation_control": (_shift_only, {"translate_x", "translate_y"},
                            "pure-translation gap; rotate must NOT dominate"),
    "decoy_pressure":     (_lowcolor, {"saturate", "hue", "bright"},
                           "recoverable chromatic gap + decoys present; read decoy leakage explicitly"),
}


# --------------------------------------------------------------------- calibration
@torch.no_grad()
def per_family_ed(visible, scorer, ae, ae_weight,
                  mags=(-1.0, -0.7, -0.4, 0.4, 0.7, 1.0), reps=2, seed=0):
    """For each family, apply at several magnitudes and report best visible->anchor ED achieved
    (lower=closer to anchor) and mean AE recon. Valid families should LOWER ED below BASE; decoys
    should not (or raise AE recon)."""
    torch.manual_seed(seed)
    base_ed = float(scorer.ed(visible))
    rows = {}
    for fam in MM.VALID_FAMILIES.keys() | MM.DECOY_FAMILIES.keys():
        best_ed, rec_at_best = base_ed, 0.0
        for mg in mags:
            eds, recs = [], []
            for _ in range(reps):
                mag = torch.full((visible.size(0),), mg, device=visible.device)
                out = MM.apply_family(fam, visible, mag)
                eds.append(float(scorer.ed(out)))
                recs.append(float(MM.ae_recon(ae, out).mean()) if ae is not None else 0.0)
            e = sum(eds) / len(eds)
            if e < best_ed:
                best_ed, rec_at_best = e, sum(recs) / len(recs)
        rows[fam] = {"best_ed": best_ed, "delta_ed": best_ed - base_ed,
                     "ae_recon": rec_at_best, "is_decoy": fam in MM.DECOY_FAMILIES}
    return base_ed, rows


def calibrate_task(name, real, seed=0, ae_steps=700):
    vis_fn, expected, note = TASKS[name]
    visible = vis_fn(real)
    scorer = MM.AnchorScorer(real, seed=777)
    ae = MM.train_robust_ae(real, steps=ae_steps, seed=seed)
    aw = 50.0
    gap = float(scorer.ed(visible))          # visible->anchor distance
    self_gap = float(scorer.ed(real[:visible.size(0)]))  # anchor->anchor floor (~0)
    base_ed, fam = per_family_ed(visible, scorer, ae, aw, seed=seed)
    # ranking by delta_ed (most negative = best at closing the gap)
    ranking = sorted(fam.items(), key=lambda kv: kv[1]["delta_ed"])
    # separability: best decoy delta vs best expected-valid delta
    valid_deltas = [fam[f]["delta_ed"] for f in expected if f in fam]
    decoy_deltas = [v["delta_ed"] for k, v in fam.items() if v["is_decoy"]]
    best_valid = min(valid_deltas) if valid_deltas else 0.0
    best_decoy = min(decoy_deltas) if decoy_deltas else 0.0
    differ = (gap - self_gap) > 0.05 * max(self_gap, 1e-6) and gap > self_gap + 0.02
    separable = best_valid < best_decoy - 1e-4   # expected valid closes gap MORE than any decoy
    return {
        "task": name, "note": note, "expected": sorted(expected),
        "visible_anchor_ed": gap, "anchor_self_ed": self_gap, "base_ed": base_ed,
        "families": fam,
        "ranking": [(k, round(v["delta_ed"], 4), v["is_decoy"]) for k, v in ranking],
        "best_valid_delta": best_valid, "best_decoy_delta": best_decoy,
        "differ_pass": bool(differ), "separable_pass": bool(separable),
        "calibration_pass": bool(differ and separable),
    }, visible, scorer, ae


# --------------------------------------------------------------------- discovery + payoff
def run_discovery(name, real, visible, scorer, ae, seed=0, steps=300, a_move=1.0, gap_aware=False):
    aw = 5.0 if gap_aware else 50.0  # downweight AE when gap-aware (reward is ED-only; AE only mild reg)
    pol = MM.MorphismPolicy(list(MM.VALID_FAMILIES) + list(MM.DECOY_FAMILIES), depth=1).to(real.device)
    print(f"[{name}] discover seed={seed} steps={steps} a_move={a_move} gap_aware={gap_aware}", flush=True)
    d = MM.discover(pol, visible, scorer, steps=steps, seed=seed, ae=ae, ae_weight=aw,
                    a_move=a_move, gap_aware=gap_aware, log_every=50)
    # rank discovered families by effective usage
    eff = d["effective_usage"]
    rank = sorted(eff.items(), key=lambda kv: -kv[1])
    return pol, d, rank


@torch.no_grad()
def payoff_ed(name, real, visible, scorer, ae, discovered_pol, seed=0):
    """Short diagnostic payoff: how much does each augmentation arm move VISIBLE toward ANCHOR (lower ED
    = better), without destroying content (AE recon). BASE = identity. ORACLE_VALID = uniform over the
    task's expected families. RANDOM_VALID = uniform over all valid. RANDOM_WITH_DECOYS = uniform over
    all (valid+decoy)."""
    torch.manual_seed(seed)
    expected = TASKS[name][1]
    vfam, dfam = list(MM.VALID_FAMILIES), list(MM.DECOY_FAMILIES)

    def uniform_pol(fams):
        p = MM.MorphismPolicy(fams, depth=1).to(real.device)
        with torch.no_grad():
            p.mag_mu.fill_(0.6)
        return p

    arms = {
        "BASE": None,
        "ORACLE_VALID": uniform_pol(sorted(expected)),
        "RANDOM_VALID": uniform_pol(vfam),
        "RANDOM_WITH_DECOYS": uniform_pol(vfam + dfam),
        "DISCOVERED": discovered_pol,
    }
    res = {}
    for arm, pol in arms.items():
        eds, recs, divs = [], [], []
        for _ in range(4):
            out = visible if pol is None else pol.sample_transform(visible)
            eds.append(float(scorer.ed(out)))
            recs.append(float(MM.ae_recon(ae, out).mean()) if ae is not None else 0.0)
            divs.append(float(out.flatten(1).std(0).mean()))
        res[arm] = {"ed": sum(eds) / len(eds), "ae_recon": sum(recs) / len(recs),
                    "diversity": sum(divs) / len(divs)}
    return res


def eqm_payoff(name, real, visible, discovered_pol, seed=0, steps=600):
    """EqM-lite payoff: train TinyEqM on the VISIBLE (gap) split with each augmentation arm, eval EqM
    velocity-matching loss on the FULL CIFAR distribution (= the held-out factor the gap removed). Lower
    eqm_full = the augmentation taught the generative model the missing factor. This is the rung between
    'moves an ED metric' and a real EqM/FID claim."""
    expected = TASKS[name][1]
    vfam, dfam = list(MM.VALID_FAMILIES), list(MM.DECOY_FAMILIES)

    def uniform_pol(fams):
        p = MM.MorphismPolicy(fams, depth=1).to(real.device)
        with torch.no_grad():
            p.mag_mu.fill_(0.6)
        return p

    arms = {
        "BASE": None,
        "ORACLE_VALID": uniform_pol(sorted(expected)),
        "RANDOM_VALID": uniform_pol(vfam),
        "RANDOM_WITH_DECOYS": uniform_pol(vfam + dfam),
        "DISCOVERED": discovered_pol,
    }
    full = real  # anchor distribution = full CIFAR (the missing-factor target)
    res = {}
    for arm, pol in arms.items():
        aug_fn = None if pol is None else (lambda x, p=pol: p.sample_transform(x))
        net = EM.train_eqm_lite(visible, aug_fn, lam=0.5, steps=steps, seed=seed)
        fc = EM.eqm_field_consistency(net, visible, full, draws=8)
        res[arm] = {"eqm_visible": fc["eqm_clean"], "eqm_full": fc["eqm_heldout"],
                    "eqm_gap": fc["eqm_gap"]}
    return res


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["calibrate", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--a_move", type=float, default=1.0)  # 0 = drop anti-identity move hinge
    ap.add_argument("--gap_aware", action="store_true")  # ED-only bandit reward (no AE veto)
    ap.add_argument("--eqm_lite", action="store_true")  # also run EqM-lite generative payoff
    ap.add_argument("--tasks", default="")  # comma list; empty=all
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    real = load_cifar(args.n, seed=args.seed)
    tasks = [t for t in (args.tasks.split(",") if args.tasks else TASKS) if t in TASKS]
    summary = {}
    for name in tasks:
        t0 = time.time()
        cal, visible, scorer, ae = calibrate_task(name, real, seed=args.seed)
        print(f"\n=== {name} === differ={cal['differ_pass']} separable={cal['separable_pass']} "
              f"gap={cal['visible_anchor_ed']:.4f} self={cal['anchor_self_ed']:.4f}", flush=True)
        print("  ranking(delta_ed, decoy):", cal["ranking"], flush=True)
        rec = {"calibration": cal}
        if args.stage == "all" and cal["calibration_pass"]:
            pol, d, rank = run_discovery(name, real, visible, scorer, ae, seed=args.seed,
                                         steps=args.steps, a_move=args.a_move, gap_aware=args.gap_aware)
            pay = payoff_ed(name, real, visible, scorer, ae, pol, seed=args.seed)
            rec["discovery"] = {"family_weights": d["family_weights"],
                                "effective_usage": d["effective_usage"],
                                "decoy_usage": d["decoy_usage"], "rank": rank}
            rec["payoff"] = pay
            print(f"  decoy_usage={d['decoy_usage']:.3f} disc_rank={rank[:4]}", flush=True)
            print(f"  payoff ED: " + ", ".join(f"{k}={v['ed']:.4f}" for k, v in pay.items()), flush=True)
            if args.eqm_lite:
                eqm = eqm_payoff(name, real, visible, pol, seed=args.seed)
                rec["eqm_lite"] = eqm
                print(f"  payoff EqM-lite (eqm_full, lower=better): "
                      + ", ".join(f"{k}={v['eqm_full']:.4f}" for k, v in eqm.items()), flush=True)
        else:
            print("  -> calibration FAILED; skipping discovery/payoff", flush=True)
        rec["seconds"] = round(time.time() - t0, 1)
        summary[name] = rec
        (OUT / f"{name}_seed{args.seed}.json").write_text(json.dumps(rec, indent=2))
        print(f"  saved {name}_seed{args.seed}.json ({rec['seconds']}s)", flush=True)
    (OUT / f"summary_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nALL DONE -> {OUT}/summary_seed{args.seed}.json", flush=True)


if __name__ == "__main__":
    main()
