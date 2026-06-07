"""v17 Phase 3 — natural-ish transfer. Does the discovered-morphism pattern hold BEYOND the synthetic
shape generator? Objects = real MNIST handwriting (colorized, placed on 32x32), with the SAME morphism
families, decoys, label-free anchor, policy, and metrics applied on top. Latent transforms are applied BY
US (known for eval only); discovery never sees them.

Arms: BASE, KNOWN_ORACLE, RANDOM_VALID, DISCOVERED_MULTI. Primary gate = EqM-lite payoff (discovered beats
random + approaches oracle, anchor essential) — the same bar Phase 2 passed on the synthetic gym.

Usage: python v17_transfer.py --task single_rotation|multi --seed 0 [--quick]
"""
import argparse
import os

import torch
import torch.nn.functional as F
import torchvision

import v17_morphism_gym as G
import v17_policy as P
import v17_eval_metrics as M
import v17_common as C

DEV = torch.device("cpu")
_MNIST = None


def _load_mnist():
    global _MNIST
    if _MNIST is None:
        d = torchvision.datasets.MNIST(os.path.expanduser("~/data"), train=True, download=False)
        x = d.data.float() / 255.0                 # [60000,28,28]
        _MNIST = (x, d.targets)
    return _MNIST


def _base_images(n, seed):
    """Colorized upright centered MNIST digits on 32x32: a real-image object manifold."""
    x, y = _load_mnist()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, x.size(0), (n,), generator=g)
    dig = x[idx]                                   # [n,28,28]
    lab = y[idx].clone()
    canvas = torch.zeros(n, 28, 28)
    canvas = dig
    mask = F.pad(canvas, (2, 2, 2, 2)).unsqueeze(1)            # [n,1,32,32] intensity in [0,1]
    # colorize: base hue (narrow), brightness, bg shade (these are the "default" appearance; the morphism
    # latents below move them for the regimes)
    hue = (torch.rand(n, generator=g) * 0.12)
    val = 0.85 * torch.ones(n)
    color = G._hsv_to_rgb(hue, torch.ones(n) * 0.9, val).view(n, 3, 1, 1)
    bg = (torch.rand(n, generator=g) * 0.10 + 0.10).view(n, 1, 1, 1)
    img = bg * (1 - mask) + color * mask
    return img.clamp(0, 1), lab


# hidden-factor configs (reuse the gym's factor indices + ranges so all metrics work unchanged)
TASKS = {
    "single_rotation": [G.ROT],
    "multi": [G.ROT, G.SCL, G.HUE],
}


def _apply_regime(img, hidden, regime, seed):
    """Apply morphism latents for the regime to base images. Returns transformed imgs + z (the applied
    offsets in the gym's 8-dim latent convention, eval-only)."""
    g = torch.Generator().manual_seed(seed)
    n = img.size(0)
    z = torch.zeros(n, G.NLAT)
    out = img
    rng = {"visible": G._VIS, "anchor": G._FULL, "heldout": G._HO}[regime]
    for f in hidden:
        fam = G.FACTOR_FAMILY[f]
        lo, hi = rng[f]
        # sample latent offset, convert to unit magnitude for the family, apply
        off = torch.rand(n, generator=g) * (hi - lo) + lo
        z[:, f] = off
        _, frange = G.VALID_FAMILIES[fam]
        unit = (off / frange).clamp(-1, 1)
        out = G.apply_family(fam, out, unit)
    return out, z


def build_transfer_bundle(task, seed, n_vis, n_anc, n_ho):
    hidden = TASKS[task]
    spec = G.TaskSpec("transfer_" + task, hidden)
    base_v, lab_v = _base_images(n_vis, seed + 1)
    base_a, lab_a = _base_images(n_anc, seed + 2)
    base_h, lab_h = _base_images(n_ho, seed + 3)
    vis, vz = _apply_regime(base_v, hidden, "visible", seed + 11)
    anc, az = _apply_regime(base_a, hidden, "anchor", seed + 12)
    ho, hz = _apply_regime(base_h, hidden, "heldout", seed + 13)
    return {"task": spec, "vis": vis, "vz": vz, "vsid": lab_v,
            "anc": anc, "az": az, "asid": lab_a, "ho": ho, "hz": hz, "hsid": lab_h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="multi", choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n_vis, n_anc, n_ho = (768, 1024, 384) if not a.quick else (512, 512, 256)
    b = build_transfer_bundle(a.task, a.seed, n_vis, n_anc, n_ho)
    scorer = C.build_scorer(b, anchor_kind="randomconv")
    vref, dref = C.ref_validity(scorer, b)
    dk = dict(steps=(150 if a.quick else 250))
    gen = torch.Generator().manual_seed(2000 + a.seed)
    steps = 250 if a.quick else 350
    torch.set_num_threads(2)

    arms = ["BASE", "KNOWN_ORACLE", "RANDOM_VALID", "DISCOVERED_MULTI", "DISCOVERED_MULTI_NO_ANCHOR"]
    res = {}
    for arm in arms:
        name = {"KNOWN_ORACLE": "KNOWN_ORACLE_MULTI", "RANDOM_VALID": "RANDOM_VALID_POLICY"}.get(arm, arm)
        pol, diag = C.make_arm(name, b, scorer=scorer, discover_kw=dk, seed=a.seed)
        em = M.train_eqm_lite(b["vis"], C.policy_aug_fn(pol), steps=steps, seed=a.seed)
        ec = M.eqm_field_consistency(em, b["vis"][:256], b["ho"][:256])
        entry = {"eqm_gap": ec["eqm_gap"], "eqm_clean": ec["eqm_clean"], "eqm_heldout": ec["eqm_heldout"]}
        if isinstance(pol, P.MorphismPolicy):
            q = M.policy_quality(pol, b["task"].true_families())
            entry["true_family_usage"] = q["true_family_usage"]; entry["decoy_usage"] = q["decoy_usage"]
            entry["recall"] = q["true_family_recall"]
        res[arm] = entry

    base, om, rv = res["BASE"]["eqm_gap"], res["KNOWN_ORACLE"]["eqm_gap"], res["RANDOM_VALID"]["eqm_gap"]
    dm, na = res["DISCOVERED_MULTI"]["eqm_gap"], res["DISCOVERED_MULTI_NO_ANCHOR"]["eqm_gap"]
    gate = {"beats_base": dm < base, "beats_random": dm < rv + 1e-3,
            "near_oracle": dm <= om + 0.05, "anchor_essential": dm < na}
    passed = all(gate.values())
    out = {"phase": "3_transfer", "dataset": "MNIST", "task": a.task, "seed": a.seed,
           "arms": res, "gate": gate, "pass": bool(passed)}
    path = C.save_json(out, "v17_transfer_%s_seed%d.json" % (a.task, a.seed))
    print("TRANSFER task=%s seed=%d PASS=%s" % (a.task, a.seed, passed))
    print("  eqm_gap: base=%.4f oracle=%.4f random=%.4f DISC=%.4f no_anchor=%.4f" % (base, om, rv, dm, na))
    if "recall" in res["DISCOVERED_MULTI"]:
        d = res["DISCOVERED_MULTI"]
        print("  DISC recall=%.2f true_use=%.2f decoy_use=%.2f" % (d["recall"], d["true_family_usage"], d["decoy_usage"]))
    print("  gate", gate, "->", path)


if __name__ == "__main__":
    main()
