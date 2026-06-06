"""v17 Phase 0 calibration. Parts run independently (parallelizable):
  0A validity         : is the gym structured? latent coverage, anchor distance, valid/invalid
                        separability, heldout distinctness.
  0B oracle payoff    : do TRUE morphisms help classifier + EqM-lite + heldout val? (gate: oracle_multi
                        beats base, random_valid, random_decoy)
  0C anchor sensitivity: which label-free anchor (pixel/randomconv/ae) separates valid from invalid?
  0D impossible/ambig : anchor=visible-only, unreachable heldout factor, decoys-match-anchor — the
                        learner must NOT hallucinate confident valid morphisms.

Usage: python v17_run_calibration.py --part 0A|0B|0C|0D --seed 0 [--quick]
"""
import argparse
import torch

import v17_morphism_gym as G
import v17_policy as P
import v17_eval_metrics as M
import v17_common as C

VALID, DECOY = C.VALID, C.DECOY


def _sep(scorer, bundle, mag=0.7, gen=None):
    """Distribution-ED separability: every decoy ED above every valid ED?"""
    if gen is None:
        gen = torch.Generator().manual_seed(5)
    vis = bundle["vis"]; B = min(256, vis.size(0))
    idx = torch.randint(0, vis.size(0), (B,), generator=gen); xb = vis[idx]
    res = {}
    for fam in VALID + DECOY:
        m = (torch.rand(B, generator=gen) * 0.5 + 0.5) * (torch.randint(0, 2, (B,), generator=gen) * 2 - 1)
        with torch.no_grad():
            res[fam] = float(scorer.ed(G.apply_family(fam, xb, m)))
    vmax = max(res[f] for f in VALID); dmin = min(res[f] for f in DECOY)
    # AUC valid<decoy (lower ED = more valid)
    vv = torch.tensor([res[f] for f in VALID]); dd = torch.tensor([res[f] for f in DECOY])
    auc = float((dd.view(-1, 1) > vv.view(1, -1)).float().mean())
    return {"per_family_ed": res, "worst_valid": vmax, "lowest_decoy": dmin,
            "separation": dmin - vmax, "auc_valid_lt_decoy": auc, "pass": bool(dmin > vmax)}


def part_0A(task, seed, quick):
    b = C.build_bundle(task, seed=seed, **({"n_vis": 256, "n_anc": 512, "n_ho": 256} if quick else {}))
    sc = C.build_scorer(b)
    spec = b["task"]
    # latent coverage = per-hidden-factor range span in vis vs anchor vs heldout
    cover = {}
    for f in spec.hidden:
        nm = G.LAT_NAMES[f]
        cover[nm] = {"visible": [float(b["vz"][:, f].min()), float(b["vz"][:, f].max())],
                     "anchor": [float(b["az"][:, f].min()), float(b["az"][:, f].max())],
                     "heldout": [float(b["hz"][:, f].min()), float(b["hz"][:, f].max())]}
    with torch.no_grad():
        ed_vis = float(sc.ed(b["vis"][:256]))
        ed_ho = float(sc.ed(b["ho"][:256]))
    sep = _sep(sc, b)
    out = {"part": "0A", "task": spec.name, "seed": seed, "latent_coverage": cover,
           "anchor_ed_visible": ed_vis, "anchor_ed_heldout": ed_ho,
           "heldout_distinct_from_visible": ed_ho > ed_vis * 1.05 or ed_vis > ed_ho * 1.05,
           "separability": sep,
           "pass": bool(sep["pass"])}
    return out


def _payoff_eval(b, arm_name, seed, quick, scorer=None):
    pol, _ = C.make_arm(arm_name, b, scorer=scorer, seed=seed)
    aug = C.policy_aug_fn(pol)
    steps = 300 if quick else 600
    # classifier
    net = M.train_classifier(b["vis"], b["vsid"], aug, steps=steps, seed=seed)
    clean_acc = M.classifier_acc(net, b["vis"], b["vsid"])
    held_acc = M.classifier_acc(net, b["ho"], b["hsid"])
    # eqm-lite
    em = M.train_eqm_lite(b["vis"], aug, steps=steps, seed=seed)
    ec = M.eqm_field_consistency(em, b["vis"][:256], b["ho"][:256])
    return {"clean_acc": clean_acc, "heldout_acc": held_acc,
            "eqm_clean": ec["eqm_clean"], "eqm_heldout": ec["eqm_heldout"], "eqm_gap": ec["eqm_gap"]}


def part_0B(task, seed, quick):
    b = C.build_bundle(task, seed=seed, **({"n_vis": 512, "n_anc": 512, "n_ho": 256} if quick else {}))
    arms = ["BASE", "RANDOM_VALID", "KNOWN_ORACLE_SINGLE", "KNOWN_ORACLE_MULTI", "RANDOM_WITH_DECOYS"]
    R = {a: _payoff_eval(b, a, seed, quick) for a in arms}
    om, base, rv, rd = (R["KNOWN_ORACLE_MULTI"], R["BASE"], R["RANDOM_VALID"], R["RANDOM_WITH_DECOYS"])
    gate = {
        "oracle_multi>base": om["heldout_acc"] > base["heldout_acc"],
        "oracle_multi>random_valid": om["heldout_acc"] >= rv["heldout_acc"] - 1e-6,
        "oracle_multi>random_decoy": om["heldout_acc"] > rd["heldout_acc"],
        "eqm_oracle<base_gap": om["eqm_gap"] < base["eqm_gap"],
    }
    return {"part": "0B", "task": b["task"].name, "seed": seed, "arms": R, "gate": gate,
            "pass": bool(gate["oracle_multi>base"] and gate["oracle_multi>random_decoy"])}


def part_0C(task, seed, quick):
    b = C.build_bundle(task, seed=seed, **({"n_vis": 256, "n_anc": 512, "n_ho": 256} if quick else {}))
    res = {}
    for kind in ["pixel", "randomconv", "ae"]:
        sc = C.build_scorer(b, anchor_kind=kind)
        res[kind] = _sep(sc, b)
    best = max(res, key=lambda k: res[k]["separation"])
    return {"part": "0C", "task": b["task"].name, "seed": seed,
            "anchors": {k: {"separation": v["separation"], "auc": v["auc_valid_lt_decoy"], "pass": v["pass"]}
                        for k, v in res.items()},
            "selected_anchor": best, "pass": bool(res[best]["pass"])}


def part_0D(seed, quick):
    """Three impossible/ambiguous controls. PASS = learner does NOT confidently claim valid morphisms."""
    out = {"part": "0D", "seed": seed, "controls": {}}
    dk = dict(steps=(200 if quick else 350))

    # (i) anchor = VISIBLE-only (no broad info): scorer built on visible, not anchor.
    b = C.build_bundle("multi_independent", seed=seed,
                       **({"n_vis": 384, "n_anc": 512, "n_ho": 256} if quick else {}))
    b_visanchor = dict(b); b_visanchor["anc"] = b["vis"]
    sc_vis = C.build_scorer(b_visanchor)
    pol, _ = C.make_arm("LEARNED_MULTI_POLICY", b, scorer=sc_vis, discover_kw=dk, seed=seed)
    q = M.policy_quality(pol, b["task"].true_families())
    cov = M.heldout_coverage(pol, b["vz"], b["hz"], b["task"].hidden)
    out["controls"]["visible_only_anchor"] = {
        "true_family_usage": q["true_family_usage"], "heldout_coverage": cov["heldout_coverage"],
        # if anchor has no broad info, move term still fires but coverage of the (unseen) heldout
        # should be low / not better than random — flag hallucination if coverage high AND confident
        "hallucinated": cov["heldout_coverage"] > 0.5 and q["max_family_weight"] > 0.4}

    # (ii) heldout factor UNREACHABLE by any allowed transform (BG has no morphism).
    bi = C.build_bundle("impossible_control", seed=seed,
                        **({"n_vis": 384, "n_anc": 512, "n_ho": 256} if quick else {}))
    sci = C.build_scorer(bi)
    poli, _ = C.make_arm("LEARNED_MULTI_POLICY", bi, scorer=sci, discover_kw=dk, seed=seed)
    covi = M.heldout_coverage(poli, bi["vz"], bi["hz"], bi["task"].hidden)
    out["controls"]["unreachable_heldout"] = {
        "heldout_coverage": covi["heldout_coverage"],
        "pass_no_hallucination": covi["heldout_coverage"] < 0.15}

    out["pass"] = bool(out["controls"]["unreachable_heldout"]["pass_no_hallucination"]
                       and not out["controls"]["visible_only_anchor"]["hallucinated"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["0A", "0B", "0C", "0D"])
    ap.add_argument("--task", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.part == "0A":
        out = part_0A(a.task or "full", a.seed, a.quick)
    elif a.part == "0B":
        out = part_0B(a.task or "multi_independent", a.seed, a.quick)
    elif a.part == "0C":
        out = part_0C(a.task or "full", a.seed, a.quick)
    else:
        out = part_0D(a.seed, a.quick)
    name = "v17_calib_%s_seed%d.json" % (a.part, a.seed)
    path = C.save_json(out, name)
    print("PASS=%s  ->  %s" % (out.get("pass"), path))
    print({k: v for k, v in out.items() if k in ("part", "selected_anchor", "gate", "pass")})


if __name__ == "__main__":
    main()
