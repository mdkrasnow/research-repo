"""v17 evaluation metrics. GROUND-TRUTH latent factors are used HERE ONLY (never in discovery/anchor).

Two metric groups:
 1. Policy-quality (label-free at train time, scored post-hoc): effective family usage, decoy usage,
    true-family recovery, entropy, collapse-to-identity, collapse-to-one-family, validity rate.
 2. GT-latent (eval only): heldout coverage, latent-factor recovery — computed analytically from each
    family's known latent action (gym.latent_delta), so we never invert images.

Also a shape classifier proxy and an EqM-lite field proxy for Phase 2 payoff.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import v17_morphism_gym as G

VALID = list(G.VALID_FAMILIES)
DECOY = list(G.DECOY_FAMILIES)


def _delta_table(families):
    """[K, NLAT] table: row j = unit latent action of family j scaled by its magnitude range (0 for decoys)."""
    T = torch.zeros(len(families), G.NLAT)
    for j, fam in enumerate(families):
        if fam in G.VALID_FAMILIES:
            _, rng = G.VALID_FAMILIES[fam]
            T[j] = G.latent_delta(fam, rng)  # delta for unit magnitude * range
    return T


# ---------------------------------------------------------------------------
# Policy-quality metrics
def effective_usage(policy):
    """w[f] * |mag_mu[f]| normalized -> how much each family actually contributes (a flat weight on a
    near-zero magnitude is NOT usage)."""
    w = policy.family_weights().detach()
    mu = policy.mag_mu.detach().abs()
    eff = w * mu
    s = eff.sum() + 1e-9
    return {f: float(e / s) for f, e in zip(policy.families, eff)}


def policy_quality(policy, true_families):
    eff = effective_usage(policy)
    decoy_use = sum(eff[f] for f in policy.families if f in DECOY)
    true_use = sum(eff.get(f, 0.0) for f in true_families)
    w = policy.family_weights().detach()
    ent = float(-(w * (w + 1e-9).log()).sum())
    maxw = float(w.max())
    # true-family recall: each true family has effective usage above uniform-ish floor
    floor = 0.5 / max(len(policy.families), 1)
    recalled = [f for f in true_families if eff.get(f, 0.0) >= floor]
    return {
        "effective_usage": eff,
        "decoy_usage": decoy_use,
        "true_family_usage": true_use,
        "true_family_recall": len(recalled) / max(len(true_families), 1),
        "true_families_recalled": recalled,
        "family_entropy": ent,
        "max_family_weight": maxw,
        "top_family": max(eff, key=eff.get) if eff else None,
    }


@torch.no_grad()
def collapse_and_validity(policy, visible, scorer, valid_ref_validity, decoy_ref_validity, n=256, gen=None):
    """move (pixel) -> identity collapse if ~0; validity rate vs a threshold midway between the
    calibration valid/decoy validity means."""
    idx = torch.randint(0, visible.size(0), (n,), generator=gen)
    xb = visible[idx]
    out = policy.sample_transform(xb)
    move_px = (out - xb).flatten(1).norm(dim=1).mean()
    val = scorer.validity(out)
    thr = 0.5 * (valid_ref_validity + decoy_ref_validity)
    return {
        "move_pixel": float(move_px),
        "identity_collapse": float(move_px) < 0.5,
        "mean_validity": float(val.mean()),
        "validity_rate": float((val > thr).float().mean()),
        "anchor_ed": float(scorer.ed(out)),
    }


# ---------------------------------------------------------------------------
# GT-latent metrics (eval only): coverage of heldout latent region by the policy's reachable set.
@torch.no_grad()
def heldout_coverage(policy, visible_z, heldout_z, hidden_factors, n_reach=4000, tol=0.12, gen=None):
    """Apply sampled policy actions to VISIBLE latents analytically (each valid family's known latent
    delta), build the reachable latent cloud, and measure what fraction of HELDOUT samples lie within
    tol of a reachable point — restricted to the hidden factors (the axes that define the task)."""
    if not getattr(policy, "families", None):
        return {"heldout_coverage": 0.0, "reachable_spread": 0.0}
    fam_idx, mag = policy.sample_actions(n_reach, gen=gen)
    base_idx = torch.randint(0, visible_z.size(0), (n_reach,), generator=gen)
    T = _delta_table(policy.families)                       # [K,NLAT]
    reach = visible_z[base_idx].clone() + mag.unsqueeze(1) * T[fam_idx]
    fdims = list(hidden_factors)
    if not fdims:
        return {"heldout_coverage": 0.0, "heldout_coverage_joint": 0.0, "reachable_spread": 0.0}
    # PER-FACTOR coverage (primary): for each hidden factor, fraction of its heldout band within tol of a
    # reachable value. Averaged over hidden factors. This is the "discovers each morphism" claim and
    # separates multi (reaches all factors) from single (reaches one) -- joint coverage is an unreasonable
    # bar (needs all factors hit simultaneously by composed single-factor moves).
    per = []
    for f in fdims:
        d1 = (heldout_z[:, f].unsqueeze(1) - reach[:, f].unsqueeze(0)).abs()
        per.append(float((d1.min(1).values < tol).float().mean()))
    cov_pf = sum(per) / len(per)
    # joint coverage (secondary, strict)
    d = torch.cdist(heldout_z[:, fdims], reach[:, fdims])
    joint = float((d.min(1).values < tol).float().mean())
    return {"heldout_coverage": cov_pf, "heldout_coverage_joint": joint,
            "per_factor_coverage": {G.LAT_NAMES[f]: per[i] for i, f in enumerate(fdims)},
            "reachable_spread": float(reach[:, fdims].std(0).mean())}


@torch.no_grad()
def latent_factor_recovery(policy, hidden_factors, n=4000, gen=None):
    """Which latent factors does the policy move, and does that match the hidden factors?
    Returns per-factor moved-variance and a recovery score (hidden moved / total moved)."""
    if not getattr(policy, "families", None):
        return {"moved_per_factor": {G.LAT_NAMES[f]: 0.0 for f in range(G.NLAT)}, "hidden_factor_mass": 0.0}
    fam_idx, mag = policy.sample_actions(n, gen=gen)
    T = _delta_table(policy.families)
    deltas = mag.unsqueeze(1) * T[fam_idx]                  # [n,NLAT]
    moved = deltas.abs().mean(0)  # [NLAT]
    total = moved.sum() + 1e-9
    hidden_mass = sum(float(moved[f]) for f in hidden_factors) / float(total)
    return {
        "moved_per_factor": {G.LAT_NAMES[f]: float(moved[f]) for f in range(G.NLAT)},
        "hidden_factor_mass": hidden_mass,
    }


# ---------------------------------------------------------------------------
# Shape classifier proxy (Phase 2 payoff)
class ShapeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, G.NSHAPE))

    def forward(self, x):
        return self.f(x)


def _aug_bank(visible, aug_fn, copies=3, seed=0):
    """Precompute `copies` augmented views of the whole visible set ONCE (chunked). Sampling from this bank
    each training step removes the per-step policy cost (the policy's grouped transform is ~0.9s/call).
    `copies`>1 keeps augmentation diversity."""
    if aug_fn is None:
        return None
    torch.manual_seed(seed + 3)
    banks = []
    for _ in range(copies):
        outs = []
        for i in range(0, visible.size(0), 256):
            outs.append(aug_fn(visible[i:i + 256]))
        banks.append(torch.cat(outs, 0))
    return torch.cat(banks, 0)  # [copies*N, ...]


def train_classifier(visible, vis_sid, aug_fn, steps=600, bs=128, lr=1e-3, seed=0, gen=None):
    torch.manual_seed(seed)
    net = ShapeNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = visible.size(0)
    g = torch.Generator().manual_seed(seed + 7)
    bank = _aug_bank(visible, aug_fn, seed=seed)
    nb = bank.size(0) if bank is not None else 0
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        if bank is not None:
            bidx = torch.randint(0, nb, (bs,), generator=g)
            xb = bank[bidx]; yb = vis_sid[bidx % n]
        else:
            xb, yb = visible[idx], vis_sid[idx]
        loss = F.cross_entropy(net(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def classifier_acc(net, imgs, sids):
    return float((net(imgs).argmax(1) == sids).float().mean())


# ---------------------------------------------------------------------------
# EqM-lite field proxy (Phase 2 payoff) — flow velocity-matching, the closest cheap analog of EqM.
class TinyEqM(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(4, 48, 3, 1, 1); self.c2 = nn.Conv2d(48, 48, 3, 1, 1)
        self.c3 = nn.Conv2d(48, 3, 3, 1, 1)

    def forward(self, x, gm):
        gch = gm.view(-1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        h = F.relu(self.c1(torch.cat([x, gch], 1))); h = F.relu(self.c2(h))
        return self.c3(h)


def eqm_loss(model, x, draws=1, gen=None):
    tot = 0.0
    for _ in range(draws):
        eps = torch.randn_like(x); gm = torch.rand(x.size(0), device=x.device)
        gg = gm.view(-1, 1, 1, 1)
        xg = (1 - gg) * x + gg * eps
        target = eps - x
        tot = tot + F.mse_loss(model(xg, gm), target)
    return tot / draws


def train_eqm_lite(visible, aug_fn, lam=0.5, steps=600, bs=128, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    net = TinyEqM(); opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = visible.size(0); g = torch.Generator().manual_seed(seed + 11)
    bank = _aug_bank(visible, aug_fn, seed=seed)
    nb = bank.size(0) if bank is not None else 0
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        xb = visible[idx]
        loss = eqm_loss(net, xb)
        if bank is not None:
            loss = loss + lam * eqm_loss(net, bank[torch.randint(0, nb, (bs,), generator=g)])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def eqm_field_consistency(model, x_clean, x_heldout, draws=8):
    """Field error on clean vs heldout-transformed validation. Robustness gap = heldout - clean."""
    torch.manual_seed(123)
    clean = float(eqm_loss(model, x_clean, draws=draws))
    held = float(eqm_loss(model, x_heldout, draws=draws))
    return {"eqm_clean": clean, "eqm_heldout": held, "eqm_gap": held - clean}
