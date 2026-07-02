"""hpsm_miner — Hard-Positive Symmetry Mining for EqM. The POSITIVE DUAL of v10 ANM.

ANM (v10): mine hard INVALID/noised examples (delta in noise space), train the field to REJECT them.
HPSM     : mine hard VALID transformed positives (named symmetry T), train the field to ACCEPT/EQUALIZE
           them — i.e. become equivariant to the valid data symmetry it currently mishandles.

    T* = argmax_T  [ hardness(T) + gap_reward(T) ]   s.t. T valid (firewall)
    L  = L_base(x) + lam_hpsm * L_eqm(T*(x)) + lam_consistency * commutator(x, T*(x))

Builds on the validated ASM miner (asm_miner.py): named family library, EqM hardness (loss / commutator /
both), validity firewall. ADDS the term that made the CPU ladder work — gap_reward (does T move x TOWARD
the anchor manifold, i.e. close a missing-factor gap) — plus entropy bonus + collapse penalty + explicit
selected_topk diagnostics. Loss-hardness ALONE ties random on full CIFAR (no gap); gap_reward is what lets
HPSM target a real missing factor (e.g. saturate on a desaturated->full gap).

Do NOT build arbitrary learned operators yet — named library only. Bottleneck = targeting + validity,
not operator expressiveness. CPU-runnable; model is a velocity field model(x_t, gamma)->velocity.
"""
from __future__ import annotations

import torch

try:
    from . import asm_miner as ASM
    from . import _multi_morphism as MM
except ImportError:  # top-level import (CPU ladder adds dganm_variants to sys.path)
    import asm_miner as ASM
    import _multi_morphism as MM

VALID = ASM.ASM_VALID            # translate_x/y, rotate, scale, hflip, pad_crop, hue, bright, contrast, saturate
DECOY = ASM.ASM_DECOY            # crop_erase, big_shear, color_collapse
ALL = ASM.ASM_ALL


def gap_reward(scorer, x, Tx):
    """How much T moves the batch TOWARD the anchor manifold (closes a gap). ED(x)-ED(Tx); positive when
    Tx is closer to anchor than x. For a desat->full gap, saturate raises chroma -> lowers ED -> high
    reward. This is the targeting signal loss-hardness lacks on a no-gap distribution."""
    with torch.no_grad():
        return float(scorer.ed(x)) - float(scorer.ed(Tx))


def commutator_consistency(model, x, name, mag):
    """Trainable equivariance loss: field-of-transform should equal transform-of-field (J_T F). Minimizing
    this teaches the EqM field to ACCEPT the valid symmetry. Differentiable (grads flow into model)."""
    B = x.size(0); dev = x.device
    gamma = torch.rand(B, device=dev) * 0.998 + 0.001
    eps = torch.randn_like(x)
    g = gamma.view(-1, 1, 1, 1)
    x_t = (1 - g) * eps + g * x
    Tx = MM.apply_family(name, x, mag)
    Tx_t = (1 - g) * eps + g * Tx
    f_x = ASM.eqm_field(model, x_t, gamma)
    f_Tx = ASM.eqm_field(model, Tx_t, gamma)
    jf = ASM.linear_action(name, f_x, mag)
    return (f_Tx - jf).pow(2).mean()


def _rms_scale(d, rms):
    """Scale per-sample so each element's RMS = rms (comparable to a real transform's O(0.5) pixel move),
    NOT a tiny fixed total-norm."""
    cur = d.flatten(1).pow(2).mean(1).sqrt().view(-1, 1, 1, 1)
    return d / (cur + 1e-8) * rms


def general_consistency(model, x, t_scale999=False, eps_radius=0.5, adv_steps=1, tangent=True):
    """GENERAL hard-positive consistency — NO named symmetry, NO hand-coded J_T. Task-agnostic for any EqM.

    Derived from the EqM target structure alone. EqM: F(x_t,g) -> target = (eps - x) (TinyEqM) or c(g)(x1-x0)
    (UNet convention). Perturb the DATA x -> x+delta (keep eps): input shifts x_t -> x_t + g*delta, target
    shifts by -delta (TinyEqM) -> so the field's RESPONSE is known exactly:
        F(x_t + g*delta, g)  should equal  sg(F(x_t,g)) - delta.
    This is a SELF-consistency (anchored on the model's own prediction sg(F), like the named commutator),
    NOT supervised augmentation. delta is mined to be (a) ON-MANIFOLD (a valid POSITIVE, not v10's
    off-manifold negative) via tangent projection orthogonal to the field, and (b) HARD (adversarially
    scaled to maximize the consistency violation). Named symmetries are a special case (delta = T(x)-x).
    """
    B = x.size(0); dev = x.device
    gamma = torch.rand(B, device=dev) * 0.998 + 0.001
    g = gamma.view(-1, 1, 1, 1)
    eps = torch.randn_like(x)
    x_t = (1 - g) * eps + g * x
    gm = (gamma * 999.0).clamp_min(0.0) if t_scale999 else gamma
    with torch.no_grad():
        F0 = model(x_t, gm)
    # on-manifold tangent: project a random perturbation orthogonal to the field (off-manifold normal ~ F)
    def _project(dd):
        if tangent:
            Fhat = F0 / (F0.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8)
            dd = dd - (dd * Fhat).flatten(1).sum(1).view(-1, 1, 1, 1) * Fhat
        return _rms_scale(dd, eps_radius)
    d = _project(torch.randn_like(x))
    # adversarial: step delta to MAXIMIZE the consistency violation (hard positive), staying on-manifold
    for _ in range(adv_steps):
        d = d.detach().requires_grad_(True)
        F1 = model(x_t + g * d, gm)
        viol = (F1 - F0 + d).pow(2).mean()
        gd = torch.autograd.grad(viol, d)[0]
        with torch.no_grad():
            d = _project(d + 0.5 * eps_radius * gd.sign())
    d = d.detach()
    F1 = model(x_t + g * d, gm)          # WITH grad into model
    return (F1 - F0.detach() + d).pow(2).mean()


def mine(model, x, scorer, ae, families=None, K=16, mode="loss_plus_comm",
         w_hard=1.0, w_gap=2.0, w_invalid=1.0, w_decoy=10.0, w_collapse=1.0, w_entropy=0.1,
         ae_weight=5.0, move_floor=0.3, mag_scale=0.8, seed=0, topk=3):
    """Mine the hardest VALID positive. score = w_hard*hardness + w_gap*gap_reward - invalid - decoy -
    collapse + entropy_bonus. Returns T_star (family, mag) + rich JSON-able diagnostics.

    Validity firewall: decoy family -> hard reject; movement < floor -> reject (trivial); magnitude bound.
    The firewall is what separates HARD-POSITIVE mining from destructive adversarial augmentation.
    """
    torch.manual_seed(seed)
    fams = families or ALL
    dev = x.device
    base_ed = float(scorer.ed(x))
    cand = []
    fam_count = {f: 0 for f in fams}
    hard_by = {f: [] for f in fams}; valid_by = {f: [] for f in fams}
    for k in range(K):
        name = fams[int(torch.randint(0, len(fams), (1,)).item())]
        fam_count[name] += 1
        mag = ((torch.rand(1) * 2 - 1) * mag_scale).to(dev).expand(x.size(0))
        Tx = MM.apply_family(name, x, mag)
        H, parts = ASM.hardness(model, x, name, mag, mode=mode)
        v = ASM.validity_penalty(scorer, ae, x, Tx, name, mag, ae_weight=ae_weight)
        gr = (base_ed - float(scorer.ed(Tx)))                       # gap_reward
        with torch.no_grad():
            div = float(Tx.flatten(1).std(0).mean())
        collapse = max(0.0, 0.05 - div)                            # collapsed-output penalty
        ent_bonus = w_entropy / (1.0 + fam_count[name])            # encourage family diversity
        is_decoy = name in DECOY
        valid = (not is_decoy) and (v["move"] >= move_floor)
        score = (w_hard * float(H) + w_gap * gr
                 - w_invalid * v["ae"] - w_decoy * v["decoy"]
                 - w_collapse * collapse - v["mag_pen"] + ent_bonus)
        cand.append({"family": name, "mag": float(mag.mean()), "hardness": float(H),
                     "gap_reward": round(gr, 4), "score": round(score, 4), "valid": valid,
                     "decoy": is_decoy, **{f"v_{kk}": round(vv, 4) for kk, vv in v.items()}, **parts})
        hard_by[name].append(float(H)); valid_by[name].append(0.0 if valid else 1.0)
    valid_cand = [c for c in cand if c["valid"]]
    valid_cand.sort(key=lambda c: -c["score"])
    top = valid_cand[:topk]
    T_star = (top[0]["family"], top[0]["mag"]) if top else ("saturate", 0.6)
    # family-level aggregates
    sel_counts = {}
    for c in top:
        sel_counts[c["family"]] = sel_counts.get(c["family"], 0) + 1
    eff = {f: sel_counts.get(f, 0) for f in fams}
    tot = sum(eff.values()) or 1
    fam_weights = {f: eff[f] / tot for f in fams}
    decoy_usage = sum(fam_weights[f] for f in DECOY)               # by construction ~0 (firewall rejects)
    diag = {
        "mode": mode, "T_star": {"family": T_star[0], "mag": round(float(T_star[1]), 4)},
        "family_weights": {f: round(v, 3) for f, v in fam_weights.items()},
        "family_counts": fam_count,
        "decoy_usage": round(decoy_usage, 4),
        "hardness_by_family": {f: round(sum(v) / len(v), 4) if v else 0.0 for f, v in hard_by.items()},
        "validity_by_family": {f: round(sum(v) / len(v), 3) if v else 1.0 for f, v in valid_by.items()},
        "selected_topk": [{"family": c["family"], "mag": round(c["mag"], 3), "score": c["score"],
                           "gap_reward": c["gap_reward"]} for c in top],
        "n_valid": len(valid_cand), "n_candidates": K,
    }
    return T_star, diag
