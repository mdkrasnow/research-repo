"""asm_miner — Adversarial Symmetry Mining for EqM (hard-POSITIVE symmetry mining).

Flip of the static v17 discovery objective. v17 asks "which transform is VALID?" (ties random on full
CIFAR — no gap, everything valid looks equal). ASM asks:

    "Which VALID transform is hardest for the CURRENT EqM model?"  (Madry-style inner maximization)

i.e. mine the data symmetry the generative vector field does not yet understand, train the field on it.
Analogue of v10 ANM, but the adversary lives in image/SYMMETRY space (valid, content-preserving) instead
of noised-perturbation space. "Hard-positive" because T(x) stays a real image of the same content.

    T* = argmax_T  Hardness_EqM(T; theta, x)   s.t.  T(x) valid, on-manifold, nontrivial, not a decoy
    theta <- min  L_base(x) + lam * L_EqM(T*(x))

Hardness modes (the EqM-native signal):
  loss_only      : H = L_EqM(T(x)) - L_EqM(x)        — field predicts worse on T(x)
  commutator_only: H = ||F(T(x)_t) - J_T F(x_t)||    — field fails to COMMUTE with the symmetry (novel)
  loss_plus_comm : a*H_loss + b*H_comm

Validity firewall (the difference vs destructive adversarial aug): anchor ED + AE recon + content
preservation + decoy-family penalty + magnitude bounds + non-identity movement. The adversary maximizes
hardness ONLY among candidates that pass validity.

Self-contained (torch + reuses _multi_morphism families/anchor/AE). CPU-runnable. The model passed in is a
velocity field model(x_t, gamma) -> velocity (same shape as x); TinyEqM (CPU ladder) or the EqM UNet (GPU).
"""
from __future__ import annotations
import math

import torch
import torch.nn.functional as F

try:
    from . import _multi_morphism as MM
except ImportError:  # loaded as a top-level module (CPU ladder adds dganm_variants to sys.path)
    import _multi_morphism as MM


# --------------------------------------------------------------------------- extra named families
def m_hflip(img, mag):
    return torch.flip(img, dims=[3])


def m_contrast(img, mag):
    mean = img.mean(dim=(2, 3), keepdim=True)
    k = (1.0 + 0.6 * mag.view(-1, 1, 1, 1))
    return (mean + k * (img - mean)).clamp(-1, 1)


def m_pad_crop(img, mag):
    # reflect-pad then shift-crop back (random-crop family). mag in [-1,1] -> shift up to +/-4 px.
    pad = 4
    xp = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    H, W = img.shape[-2], img.shape[-1]
    dx = int(round((mag.mean().item() * 0.5 + 0.5) * 2 * pad))
    dy = dx
    dx = max(0, min(2 * pad, dx)); dy = max(0, min(2 * pad, dy))
    return xp[:, :, dy:dy + H, dx:dx + W]


# register additive families into MM dicts (idempotent; bridge defaults already include saturate)
MM.VALID_FAMILIES.setdefault("hflip", (m_hflip, 1.0))
MM.VALID_FAMILIES.setdefault("contrast", (m_contrast, 1.0))
MM.VALID_FAMILIES.setdefault("pad_crop", (m_pad_crop, 1.0))
if "saturate" not in MM.VALID_FAMILIES:
    def _m_sat(img, mag):
        g = img.mean(1, keepdim=True)
        return (g + (1.0 + mag.view(-1, 1, 1, 1)) * (img - g)).clamp(-1, 1)
    MM.VALID_FAMILIES["saturate"] = (_m_sat, 0.8)

ASM_VALID = ["translate_x", "translate_y", "rotate", "scale", "hflip", "pad_crop",
             "hue", "bright", "contrast", "saturate"]
ASM_DECOY = list(MM.DECOY_FAMILIES)
ASM_ALL = ASM_VALID + ASM_DECOY


# --------------------------------------------------------------------------- linear action J_T on a field
def _color_hue_M(mag, device):
    ang = mag * 2 * math.pi
    cos, sin = torch.cos(ang), torch.sin(ang)
    kx = 1.0 / math.sqrt(3.0); oc = (1 - cos) / 3.0
    a = cos + oc; b = oc - kx * sin; c = oc + kx * sin
    return torch.stack([torch.stack([a, b, c], 1), torch.stack([c, a, b], 1),
                        torch.stack([b, c, a], 1)], 1)  # [B,3,3]


def linear_action(name, field, mag):
    """J_T applied to a velocity FIELD (same shape as image). For a symmetry T acting on image space, the
    field of T(x) should equal J_T applied to the field of x. Spatial T -> warp the field by the SAME grid;
    color-matrix T -> apply matrix to field; bright (additive) -> identity on field; contrast/saturate ->
    their linear part about the field's own mean/gray."""
    B = field.size(0)
    if name in ("translate_x", "translate_y", "rotate", "scale", "hflip", "pad_crop"):
        # warp field spatially the same way the image is warped (channels carry along)
        return MM.apply_family(name, field, mag) if name in MM.VALID_FAMILIES else field
    if name == "hue":
        M = _color_hue_M(mag * 0.5, field.device)  # rng matches m_hue (0.5)
        return torch.bmm(M, field.view(B, 3, -1)).view_as(field)
    if name == "bright":
        return field  # additive shift -> J = identity on the velocity
    if name == "contrast":
        mean = field.mean(dim=(2, 3), keepdim=True)
        k = (1.0 + 0.6 * mag.view(-1, 1, 1, 1))
        return mean + k * (field - mean)
    if name == "saturate":
        g = field.mean(1, keepdim=True)
        return g + (1.0 + (mag * 0.8).view(-1, 1, 1, 1)) * (field - g)
    return field  # decoys: no defined linear action (they aren't symmetries)


# --------------------------------------------------------------------------- EqM forward + hardness
def _noise_mix(x, gamma, eps=None):
    if eps is None:
        eps = torch.randn_like(x)
    g = gamma.view(-1, 1, 1, 1)
    return (1 - g) * eps + g * x, eps


def eqm_field(model, x, gamma):
    # model(x_t, gamma_scaled) -> velocity. gamma in (0,1); scale to model's expected t.
    return model(x, (gamma * 999.0).clamp_min(0.0) if _wants_big_t(model) else gamma)


def _wants_big_t(model):
    # TinyEqM(v17) takes gamma in (0,1); EqM UNet takes t~[0,999]. Heuristic flag set by caller.
    return getattr(model, "_t_scale_999", False)


def eqm_loss_on(model, x, gamma=None, eps=None):
    B = x.size(0); dev = x.device
    if gamma is None:
        gamma = torch.rand(B, device=dev) * 0.998 + 0.001
    x_t, eps = _noise_mix(x, gamma, eps)
    target = eps - x  # flow velocity target (matches v17 TinyEqM convention)
    pred = eqm_field(model, x_t, gamma)
    return F.mse_loss(pred, target), gamma, eps


def hardness(model, x, name, mag, mode="loss_plus_comm", a=1.0, b=1.0):
    """Return (H, parts). Uses a SHARED (gamma, eps) draw for x and T(x) so the comparison is paired."""
    B = x.size(0); dev = x.device
    gamma = torch.rand(B, device=dev) * 0.998 + 0.001
    eps = torch.randn_like(x)
    Tx = MM.apply_family(name, x, mag)
    parts = {}
    H = torch.zeros((), device=dev)
    if mode in ("loss_only", "loss_plus_comm"):
        l_base, _, _ = eqm_loss_on(model, x, gamma, eps)
        l_T, _, _ = eqm_loss_on(model, Tx, gamma, eps)
        h_loss = (l_T - l_base)
        parts["h_loss"] = float(h_loss)
        H = H + (a * h_loss if mode == "loss_plus_comm" else h_loss)
    if mode in ("commutator_only", "loss_plus_comm"):
        x_t, _ = _noise_mix(x, gamma, eps)
        Tx_t, _ = _noise_mix(Tx, gamma, eps)
        f_x = eqm_field(model, x_t, gamma)
        f_Tx = eqm_field(model, Tx_t, gamma)
        jf = linear_action(name, f_x, mag)
        h_comm = (f_Tx - jf).flatten(1).norm(dim=1).mean()
        parts["h_comm"] = float(h_comm)
        H = H + (b * h_comm if mode == "loss_plus_comm" else h_comm)
    return H, parts


# --------------------------------------------------------------------------- validity firewall
def validity_penalty(scorer, ae, x, Tx, name, mag, ae_weight=5.0):
    """Lower = more valid. Combines on-manifold ED (vs anchor ref), AE recon (off-manifold), content
    drift, decoy-family flag, magnitude bound, and an anti-identity (movement) requirement."""
    with torch.no_grad():
        ed = float(scorer.ed(Tx))                                   # distance to anchor manifold
        ae_rec = float(MM.ae_recon(ae, Tx).mean()) * ae_weight if ae is not None else 0.0
        content = float((Tx - x).flatten(1).norm(dim=1).mean())     # content drift (also = movement)
        decoy = 1.0 if name in ASM_DECOY else 0.0
        magabs = float(mag.abs().mean())
    mag_pen = max(0.0, magabs - 1.0) ** 2
    move = content
    return {"ed": ed, "ae": ae_rec, "content": content, "decoy": decoy,
            "mag_pen": mag_pen, "move": move}


# --------------------------------------------------------------------------- ASM inner loop
@torch.no_grad()
def _sample_mag(name, scale=0.8):
    return (torch.rand(1) * 2 - 1) * scale


def mine(model, x, scorer, ae, families=None, K=32, mode="loss_plus_comm",
         a=1.0, b=1.0, ae_weight=5.0,
         w_ed=1.0, w_ae=1.0, w_content=0.0, w_decoy=10.0, w_move=0.5, move_floor=0.3,
         topm=2, seed=0):
    """Sample K candidate (family, magnitude) transforms, score each by HARDNESS minus VALIDITY penalties,
    keep only valid ones, return the top-M hard-valid transforms + per-family diagnostics.

    Score(T) = H(T) - w_ed*ED - w_ae*AE - w_content*content - w_decoy*decoy - w_move*max(0, move_floor-move)
    """
    torch.manual_seed(seed)
    fams = families or ASM_ALL
    cand = []
    by_fam_hard = {f: [] for f in fams}
    by_fam_valid = {f: [] for f in fams}
    for k in range(K):
        name = fams[torch.randint(0, len(fams), (1,)).item()]
        mag = _sample_mag(name).to(x.device).expand(x.size(0))
        Tx = MM.apply_family(name, x, mag)
        H, parts = hardness(model, x, name, mag, mode=mode, a=a, b=b)
        v = validity_penalty(scorer, ae, x, Tx, name, mag, ae_weight=ae_weight)
        # validity gate: reject decoys, off-manifold (AE), or trivial (no movement)
        valid = (v["decoy"] == 0.0) and (v["move"] >= move_floor)
        score = (float(H)
                 - w_ed * v["ed"] - w_ae * v["ae"] - w_content * v["content"]
                 - w_decoy * v["decoy"] - w_move * max(0.0, move_floor - v["move"]) - v["mag_pen"])
        rec = {"family": name, "mag": float(mag.mean()), "H": float(H), "score": score,
               "valid": valid, **{f"v_{kk}": vv for kk, vv in v.items()}, **parts}
        cand.append(rec)
        by_fam_hard[name].append(float(H))
        by_fam_valid[name].append(0.0 if valid else 1.0)
    valid_cand = [c for c in cand if c["valid"]]
    valid_cand.sort(key=lambda c: -c["score"])
    top = valid_cand[:topm]
    # family-level aggregates
    hard_by_family = {f: (sum(v) / len(v) if v else 0.0) for f, v in by_fam_hard.items()}
    invalid_rate = {f: (sum(v) / len(v) if v else 0.0) for f, v in by_fam_valid.items()}
    n_valid = len(valid_cand)
    decoy_in_top = sum(1 for c in top if c["family"] in ASM_DECOY)
    return {
        "top": top,
        "n_candidates": K, "n_valid": n_valid,
        "decoy_in_top": decoy_in_top,
        "hardness_by_family": {f: round(v, 4) for f, v in hard_by_family.items()},
        "invalid_rate_by_family": {f: round(v, 3) for f, v in invalid_rate.items()},
        "top_families": [c["family"] for c in top],
        "all": cand,
    }
