"""v17 Phase 1 discovery. One task x seed -> all arms + ablations, scored on policy-quality + GT-latent
coverage (eval only). Gates are evaluated by v17_collect_results.py across seeds.

Arms: BASE_IDENTITY, KNOWN_ORACLE, RANDOM_VALID_POLICY, RANDOM_WITH_DECOYS,
      LEARNED_SINGLE_POLICY, LEARNED_MULTI_POLICY,
      LEARNED_MULTI_NO_ANCHOR, LEARNED_MULTI_NO_DIVERSITY, LEARNED_MULTI_NO_BOUNDS.

Usage: python v17_run_discovery.py --task single_rotation --seed 0 [--anchor randomconv] [--quick]
"""
import argparse
import torch

import v17_policy as P
import v17_eval_metrics as M
import v17_common as C

ARMS = ["BASE_IDENTITY", "KNOWN_ORACLE", "RANDOM_VALID_POLICY", "RANDOM_WITH_DECOYS",
        "LEARNED_SINGLE_POLICY", "LEARNED_MULTI_POLICY",
        "LEARNED_MULTI_NO_ANCHOR", "LEARNED_MULTI_NO_DIVERSITY", "LEARNED_MULTI_NO_BOUNDS"]


def eval_arm(pol, diag, b, scorer, vref, dref, gen):
    spec = b["task"]
    out = {}
    if isinstance(pol, P.MorphismPolicy):
        out["quality"] = M.policy_quality(pol, spec.true_families())
        if diag is not None:
            out["discover_final"] = diag.get("final", {})
    out["collapse_validity"] = M.collapse_and_validity(pol, b["vis"], scorer, vref, dref, gen=gen)
    out["coverage"] = M.heldout_coverage(pol, b["vz"], b["hz"], spec.hidden, gen=gen)
    out["latent_recovery"] = M.latent_factor_recovery(pol, spec.hidden, gen=gen)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor", default="randomconv")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    szkw = {"n_vis": 512, "n_anc": 768, "n_ho": 384} if a.quick else {}
    b = C.build_bundle(a.task, seed=a.seed, **szkw)
    scorer = C.build_scorer(b, anchor_kind=a.anchor)
    vref, dref = C.ref_validity(scorer, b)
    dk = dict(steps=(150 if a.quick else 250))
    gen = torch.Generator().manual_seed(1000 + a.seed)

    results = {}
    for arm in ARMS:
        pol, diag = C.make_arm(arm, b, scorer=scorer, discover_kw=dk, seed=a.seed)
        results[arm] = eval_arm(pol, diag, b, scorer, vref, dref, gen)

    out = {"task": b["task"].name, "seed": a.seed, "anchor": a.anchor,
           "true_families": b["task"].true_families(), "hidden": b["task"].hidden,
           "valid_ref_validity": vref, "decoy_ref_validity": dref, "arms": results}
    name = "v17_discovery_%s_seed%d.json" % (a.task, a.seed)
    path = C.save_json(out, name)

    # quick console summary
    print("task=%s seed=%d anchor=%s true=%s" % (a.task, a.seed, a.anchor, out["true_families"]))
    for arm in ARMS:
        r = results[arm]
        cov = r["coverage"]["heldout_coverage"]
        val = r["collapse_validity"]["validity_rate"]
        du = r["quality"]["decoy_usage"] if "quality" in r else float("nan")
        tu = r["quality"]["true_family_usage"] if "quality" in r else float("nan")
        print("  %-26s cov=%.3f val=%.2f decoy_use=%.2f true_use=%.2f" % (arm, cov, val, du, tu))
    print("->", path)


if __name__ == "__main__":
    main()
