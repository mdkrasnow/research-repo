"""
Exact forward-over-reverse parameter gradient for ebm='direct' training
(2026-08-07, post-Gate-2 theoretical redesign).

Background: fb_direct's semigradient (dropping dL/dC * dC/dtheta) was proven
biased O(1) (Gate 1 cosine 0.62) and non-conservative -- Gate 2 diverged
smoothly (10.85 -> 22.62) with moderate gradient norms, the signature of
flowing along a non-symmetric-Jacobian update field that is not the gradient
of any potential. The repair implemented here restores EXACTNESS instead of
managing the bias:

Pearlmutter (1994) forward-over-reverse. For the training loss

    L(theta) = mean_elem( (f - u)^2 ),     f = -grad_z E_theta(z)

write g := grad_z E, f = -g. Then with per-element weight 1/(B*D):

    dL/dtheta = (2/(B*D)) * sum (f - u) * df/dtheta
              = (2/(B*D)) * (g + u)^T * d(grad_z E)/dtheta
              = d/dtheta [ w^T grad_z E ]                    (w detached)
              = d/dtheta [ D_w E ]                           (directional deriv)

where w = (2/(B*D)) * (g + u). D_w E is a Jacobian-vector product in the
INPUT z, computable by forward-mode AD (torch.autograd.forward_ad dual
tensors) with NO backward graph over a backward pass; one ordinary
first-order reverse pass over the scalar sum(D_w E) w.r.t. theta then yields
the exact mixed term v^T d^2E/(dz dtheta) -- the same quantity ebm='direct'
gets from create_graph=True double-backward, at ~2x forward cost and
ordinary-training memory, with no double-backward graph retention.

Optional gradient penalty (smoothness regularizer on the energy landscape,
targeting the conditioning of the mixed Hessian term at the source):

    L_gp = gp_lambda * mean_elem( g^2 )
    dL_gp/dtheta = (2*gp_lambda/(B*D)) * g^T * d(grad_z E)/dtheta

which folds into the SAME single JVP direction (linearity):

    w_total = (2/(B*D)) * [ (g + u) + gp_lambda * g ]

so the exact gradient of L + L_gp still costs exactly one dual forward pass
plus one ordinary backward.

Exactness caveat (label dropout): EqM's y_embedder applies CFG label dropout
when model.training is True. This module runs TWO forward passes (one
first-order pass to get g and the loss; one dual pass for the HVP); an
independent dropout draw per pass would silently break the identity above.
exact_fwrev_backward therefore pre-draws the dropout ONCE via
y_embedder.token_drop and temporarily zeroes y_embedder.dropout_prob for the
duration of both passes, restoring it afterwards. Verified equivalent to the
single-pass double-backward gradient in tests/test_fb_direct_exact_hvp.py.
"""
import contextlib

import torch
import torch.autograd.forward_ad as fwAD

from transport.utils import mean_flat


@contextlib.contextmanager
def _forward_ad_safe_ops():
    """torch 2.7.1's composite-kernel forward-AD formulas break
    forward-over-reverse in two distinct ways, both found by minimal repro
    and both fixed by substituting compositions of primitive ops (matmul /
    exp / tanh / gelu / silu / sigmoid all compose cleanly):

    1. softmax / logsumexp: their fwAD formulas perform an in-place
       mutation that poisons the combined graph ('modified by an inplace
       operation ... output 0 of ExpBackward0'). Substitute
       exp(x - max(x).detach()) / sum(...) -- mathematically identical
       (softmax's shift invariance makes the detached max subtraction exact
       for value and derivative). Patched at torch.Tensor.softmax so timm's
       manual attention path (`attn.softmax(dim=-1)`) picks it up.

    2. native_layer_norm: its fwAD formula treats the saved mean/rstd as
       CONSTANTS, silently dropping the dependence of the LN Jacobian on
       upstream parameters -- tangent VALUES are exact, but
       grad_theta(tangent) is systematically wrong (microtest: 5.7e0 error
       through a matmul-LN-modulate chain; manual mean/var composition:
       1.3e-15). This is a silent-wrong-answer bug, not a crash, and it
       lands exactly on the adaLN-conditioned LN sites at every SiT block.
       Patched at torch.nn.functional.layer_norm (nn.LayerNorm.forward
       routes through it)."""
    orig_softmax = torch.Tensor.softmax
    orig_layer_norm = torch.nn.functional.layer_norm

    def oop_softmax(self, dim=-1, dtype=None):
        x = self if dtype is None else self.to(dtype)
        x = x - x.amax(dim=dim, keepdim=True).detach()
        e = x.exp()
        return e / e.sum(dim=dim, keepdim=True)

    def oop_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        dims = tuple(range(input.dim() - len(normalized_shape), input.dim()))
        mu = input.mean(dim=dims, keepdim=True)
        var = ((input - mu) ** 2).mean(dim=dims, keepdim=True)
        out = (input - mu) / (var + eps).sqrt()
        if weight is not None:
            out = out * weight
        if bias is not None:
            out = out + bias
        return out

    torch.Tensor.softmax = oop_softmax
    torch.nn.functional.layer_norm = oop_layer_norm
    try:
        yield
    finally:
        torch.Tensor.softmax = orig_softmax
        torch.nn.functional.layer_norm = orig_layer_norm


def _unfused_attention(model):
    """Temporarily force timm Attention modules onto their manual
    (q@k -> softmax -> @v) path. The fused F.scaled_dot_product_attention
    MATH-backend composite contains an in-place op on the softmax's exp
    output that breaks the combined forward-AD + reverse-AD graph
    ('modified by an inplace operation ... output 0 of ExpBackward0').
    The manual path is numerically the math backend's algorithm (same one
    train.py already forces for all ebm != 'none' training) implemented
    with out-of-place ops. Forcing it for BOTH internal passes also keeps
    pass-1 (g, loss) and pass-2 (dual) kernel-consistent.
    Returns restore_fn."""
    flipped = []
    for m in model.modules():
        if hasattr(m, "fused_attn") and m.fused_attn:
            m.fused_attn = False
            flipped.append(m)
    def restore():
        for m in flipped:
            m.fused_attn = True
    return restore


def _predrop_labels(model, y):
    """Draw CFG label dropout once, so both forward passes see identical
    labels. Returns (dropped_y, restore_fn)."""
    ye = model.y_embedder
    p_drop = float(getattr(ye, "dropout_prob", 0.0))
    if model.training and p_drop > 0:
        with torch.no_grad():
            y = ye.token_drop(y)
    ye.dropout_prob = 0.0
    def restore():
        ye.dropout_prob = p_drop
    return y, restore


def exact_fwrev_backward(model, xt, t, y, ut, gp_lambda=0.0):
    """Computes the EXACT gradient of

        L = mean_flat((-grad_z E - ut)**2).mean()
            + gp_lambda * mean_flat((grad_z E)**2).mean()

    w.r.t. model parameters WITHOUT create_graph=True double-backward, and
    ACCUMULATES it into each parameter's .grad (mirroring loss.backward()
    semantics so the caller's existing clip/opt.step/logging path is
    unchanged).

    model: raw EqM module (NOT DDP-wrapped) with ebm='direct'.
    Returns dict: loss_main (float), loss_gp (float), field_norm, target_norm.
    """
    if getattr(model, "ebm", None) != "direct":
        raise ValueError(f"exact_fwrev_backward requires ebm='direct', got {getattr(model, 'ebm', None)!r}")

    y_dropped, restore_labels = _predrop_labels(model, y)
    restore_attn = _unfused_attention(model)
    try:
      with _forward_ad_safe_ops():
        B = xt.shape[0]
        D = xt[0].numel()

        # ---- Pass 1 (ordinary first-order): g = grad_z E, loss values ----
        z = xt.detach().clone().requires_grad_(True)
        E = model(z, t, y_dropped, energy_only=True)
        g = torch.autograd.grad(E.sum(), z, create_graph=False)[0].detach()
        field = -g
        loss_main = mean_flat((field - ut) ** 2).mean()
        loss_gp = mean_flat(g ** 2).mean()

        # ---- Combined JVP direction (see module docstring derivation) ----
        w = (2.0 / (B * D)) * ((g + ut) + gp_lambda * g)
        w = w.detach()

        # ---- Pass 2 (dual forward + one ordinary reverse over theta) ----
        with fwAD.dual_level():
            z_dual = fwAD.make_dual(xt.detach().clone(), w)
            E_dual = model(z_dual, t, y_dropped, energy_only=True)
            _, tangent = fwAD.unpack_dual(E_dual)
            if tangent is None:
                raise RuntimeError(
                    "forward-mode AD produced no tangent through the energy head -- "
                    "an op in the model lacks forward-AD support; do not silently "
                    "fall back (that would reintroduce the semigradient bias)."
                )
            s = tangent.sum()
        s.backward()  # accumulates exact dL/dtheta into p.grad, first-order only
    finally:
        restore_attn()
        restore_labels()

    return {
        "loss_main": float(loss_main.detach()),
        "loss_gp": float(loss_gp.detach()),
        "field_norm": float(field.detach().norm()),
        "target_norm": float(ut.detach().norm()),
    }
