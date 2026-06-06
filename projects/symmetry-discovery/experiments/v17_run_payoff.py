"""v17 Phase 2 payoff. Do DISCOVERED morphisms improve a downstream model on the HELDOUT region?
Two proxies (run separately, parallelizable): classifier and eqm_lite.

Arms: BASE, KNOWN_ORACLE_MULTI, RANDOM_VALID_POLICY, DISCOVERED_SINGLE, DISCOVERED_MULTI,
      DISCOVERED_MULTI_NO_ANCHOR, DISCOVERED_MULTI_NO_DIVERSITY.

Classifier: clean acc, heldout acc (shapes rendered in the held region), robustness gap (clean-heldout).
EqM-lite : clean field loss, heldout field loss, robustness gap.

Gate (collector): discovered_multi > base, > random_valid, > discovered_single, approaches oracle, and
EqM-lite improves (not only classifier).

Usage: python v17_run_payoff.py --task multi_independent --seed 0 --proxy classifier|eqm_lite [--quick]
"""
import argparse
import torch

import v17_eval_metrics as M
import v17_common as C

ARMS = ["BASE", "KNOWN_ORACLE_MULTI", "RANDOM_VALID_POLICY", "DISCOVERED_SINGLE", "DISCOVERED_MULTI",
        "DISCOVERED_MULTI_NO_ANCHOR", "DISCOVERED_MULTI_NO_DIVERSITY"]


def classifier_payoff(b, pol, seed, steps):
    aug = C.policy_aug_fn(pol)
    net = M.train_classifier(b["vis"], b["vsid"], aug, steps=steps, seed=seed)
    clean = M.classifier_acc(net, b["vis"], b["vsid"])
    held = M.classifier_acc(net, b["ho"], b["hsid"])
    return {"clean_acc": clean, "heldout_acc": held, "robustness_gap": clean - held}


def eqm_payoff(b, pol, seed, steps):
    aug = C.policy_aug_fn(pol)
    em = M.train_eqm_lite(b["vis"], aug, steps=steps, seed=seed)
    ec = M.eqm_field_consistency(em, b["vis"][:256], b["ho"][:256])
    return {"eqm_clean": ec["eqm_clean"], "eqm_heldout": ec["eqm_heldout"], "robustness_gap": ec["eqm_gap"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proxy", required=True, choices=["classifier", "eqm_lite"])
    ap.add_argument("--anchor", default="randomconv")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    szkw = {"n_vis": 512, "n_anc": 768, "n_ho": 384} if a.quick else {}
    b = C.build_bundle(a.task, seed=a.seed, **szkw)
    scorer = C.build_scorer(b, anchor_kind=a.anchor)
    dk = dict(steps=(250 if a.quick else 450))
    steps = 350 if a.quick else 700

    results = {}
    for arm in ARMS:
        pol, _ = C.make_arm(arm, b, scorer=scorer, discover_kw=dk, seed=a.seed)
        if a.proxy == "classifier":
            results[arm] = classifier_payoff(b, pol, a.seed, steps)
        else:
            results[arm] = eqm_payoff(b, pol, a.seed, steps)

    out = {"task": b["task"].name, "seed": a.seed, "proxy": a.proxy, "anchor": a.anchor, "arms": results}
    name = "v17_payoff_%s_%s_seed%d.json" % (a.task, a.proxy, a.seed)
    path = C.save_json(out, name)
    key = "heldout_acc" if a.proxy == "classifier" else "robustness_gap"
    better = "higher" if a.proxy == "classifier" else "lower"
    print("task=%s proxy=%s seed=%d (%s=%s is better)" % (a.task, a.proxy, a.seed, key, better))
    for arm in ARMS:
        print("  %-30s %s=%.4f" % (arm, key, results[arm][key]))
    print("->", path)


if __name__ == "__main__":
    main()
