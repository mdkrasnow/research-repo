"""v17 shared harness: dataset bundles, anchor scorer, arm constructors, calibration, JSON IO.

Keeps the run_* scripts thin/config-driven (one ladder, not many one-offs). Imported by every v17_run_*.
"""
import json
import os

import torch

import v17_morphism_gym as G
import v17_policy as P
import v17_eval_metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
VALID = list(G.VALID_FAMILIES)
DECOY = list(G.DECOY_FAMILIES)
ALL_FAMILIES = VALID + DECOY


def build_bundle(task, n_vis=1024, n_anc=1536, n_ho=512, seed=0):
    spec = G.TASKS[task] if isinstance(task, str) else task
    vis, vz, vsid = G.make_dataset(n_vis, "visible", seed=seed + 1, task=spec)
    anc, az, asid = G.make_dataset(n_anc, "anchor", seed=seed + 2, task=spec)
    ho, hz, hsid = G.make_dataset(n_ho, "heldout", seed=seed + 3, task=spec)
    return {"task": spec, "vis": vis, "vz": vz, "vsid": vsid,
            "anc": anc, "az": az, "asid": asid, "ho": ho, "hz": hz, "hsid": hsid}


def build_scorer(bundle, anchor_kind="randomconv", seed=777, **kw):
    enc = G.build_anchor(anchor_kind, bundle["anc"], seed=seed)
    return G.AnchorScorer(enc, bundle["anc"], **kw)


def ref_validity(scorer, bundle, mag=0.7, gen=None):
    """Mean per-image validity of valid-transformed vs decoy-transformed visible (calibrates threshold)."""
    if gen is None:
        gen = torch.Generator().manual_seed(99)
    vis = bundle["vis"]
    B = min(256, vis.size(0))
    idx = torch.randint(0, vis.size(0), (B,), generator=gen)
    xb = vis[idx]

    def mean_val(fams):
        vs = []
        for f in fams:
            m = (torch.rand(B, generator=gen) * 0.5 + 0.5) * (torch.randint(0, 2, (B,), generator=gen) * 2 - 1)
            vs.append(scorer.validity(G.apply_family(f, xb, m)))
        return float(torch.cat(vs).mean())

    return mean_val(VALID), mean_val(DECOY)


# ---------------------------------------------------------------------------
# Arms: each returns an object with .sample_transform(img) (and we wrap to aug_fn for payoff).
def make_arm(name, bundle, scorer=None, discover_kw=None, seed=0):
    spec = bundle["task"]
    true_f = spec.true_families()
    depth = 2 if "composed" in spec.name else 1
    if name in ("BASE", "BASE_IDENTITY"):
        return P.FixedPolicy([], identity=True), None
    if name in ("KNOWN_ORACLE", "KNOWN_ORACLE_MULTI"):
        return P.FixedPolicy(true_f or VALID, depth=depth), None
    if name == "KNOWN_ORACLE_SINGLE":
        return P.FixedPolicy(true_f[:1] or VALID[:1], depth=1), None
    if name in ("RANDOM_VALID", "RANDOM_VALID_POLICY"):
        return P.FixedPolicy(VALID, depth=depth), None
    if name in ("RANDOM_WITH_DECOYS",):
        return P.FixedPolicy(ALL_FAMILIES, depth=depth), None
    # learned policies
    learned = {
        "LEARNED_SINGLE_POLICY": dict(depth=1),
        "LEARNED_MULTI_POLICY": dict(depth=depth),
        "LEARNED_MULTI_NO_ANCHOR": dict(depth=depth, use_anchor=False),
        "LEARNED_MULTI_NO_DIVERSITY": dict(depth=depth, use_diversity=False),
        "LEARNED_MULTI_NO_BOUNDS": dict(depth=depth, use_bounds=False),
        # payoff aliases
        "DISCOVERED_SINGLE": dict(depth=1),
        "DISCOVERED_MULTI": dict(depth=depth),
        "DISCOVERED_MULTI_NO_ANCHOR": dict(depth=depth, use_anchor=False),
        "DISCOVERED_MULTI_NO_DIVERSITY": dict(depth=depth, use_diversity=False),
    }
    if name not in learned:
        raise ValueError("unknown arm " + name)
    cfg = dict(learned[name])
    pol_depth = cfg.pop("depth")
    pol = P.MorphismPolicy(ALL_FAMILIES, depth=pol_depth)
    kw = dict(discover_kw or {})
    kw.update(cfg)
    diag = P.discover(pol, bundle["vis"], scorer, seed=seed, **kw)
    if name in ("LEARNED_SINGLE_POLICY", "DISCOVERED_SINGLE"):
        # collapse to the single best discovered morphism (tests "one morphism is insufficient for multi")
        with torch.no_grad():
            top = int(pol.logits.argmax())
            new = torch.full_like(pol.logits, -1e9)
            new[top] = 0.0
            pol.logits.copy_(new)
        if diag is not None:
            diag["collapsed_to"] = pol.families[top]
    return pol, diag


def policy_aug_fn(policy):
    g = torch.Generator().manual_seed(0)

    def f(x):
        if isinstance(policy, P.FixedPolicy):
            return policy.sample_transform(x, gen=g)
        return policy.sample_transform(x)
    return f


# ---------------------------------------------------------------------------
def save_json(obj, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    return path


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as fh:
        return json.load(fh)
