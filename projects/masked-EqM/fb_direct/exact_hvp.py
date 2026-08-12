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


def allreduce_fwrev_grads(module, process_group=None):
    """Average every parameter's .grad across ranks with ONE flat all_reduce.

    exact-fwrev bypasses loss.backward(), so DDP's bucketed allreduce hooks
    never fire; this is the explicit replacement, matching DDP's averaging
    semantics (SUM then divide by world size). Ranks stay synchronized by
    the same argument as fb_direct's original DDP design: identical
    post-allreduce gradients + identical optimizer state (same checkpoint,
    deterministic AdamW) + identical clip threshold => identical parameter
    trajectories on every rank. Coalesced into a single flat tensor so the
    ~130 parameter tensors cost one collective per step, not 130.

    No-op when torch.distributed is unavailable/uninitialized or ws == 1.
    """
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return
    ws = dist.get_world_size(process_group)
    if ws == 1:
        return
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat, group=process_group)
    flat.div_(ws)
    for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)


def fwrev_rank_sync_checksum(module):
    """Cheap cross-rank desync detector: a scalar checksum of the parameter
    state. Callers all_reduce MIN and MAX of this and compare -- identical
    ranks give spread 0. float64 so the check isn't noise-limited."""
    with torch.no_grad():
        return torch.stack(
            [p.detach().double().mean() for p in module.parameters()]
        ).sum()


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
        # ||w||: the real residual/JVP direction magnitude (2026-08-10 spike
        # instrumentation) -- w is already computed above, this is a free
        # by-product, not an extra pass. Lets a caller form the
        # residual-CONDITIONED amplification ratio grad_norm/w_norm, instead
        # of the isotropic-random-direction U_t that field_sensitivity_
        # diagnostic.py used (which came back wrong-signed -- a mean/
        # isotropic quantity blind to the tail direction the real training
        # dynamics actually push along).
        "w_norm": float(w.detach().norm()),
    }


def exact_field_vjp(model, xt, t, y, v):
    """Exact VJP of the FIELD f_theta(x) = -grad_z E_theta(x) w.r.t. theta,
    in an ARBITRARY caller-supplied direction v (same shape as xt/field) --
    generalizes exact_fwrev_backward's internally fixed direction
    w = (2/(B*D))*[(g+u) + gp_lambda*g] to any v. This is what
    matched_replay_jacobian_diagnostic.py needs: the CANONICAL,
    unreduced/unrescaled residual-conditioned direction v = r/||r||_2 (r =
    field - target, no 1/(B*D) factor), not the training loss's specific
    reduction convention baked into w. (Phase 0 audit, 2026-08-10: using
    w_norm as a stand-in for ||r|| in spike_batch_amplification_diagnostic.py
    was exactly this bug -- w is residual rescaled by 2/(B*D) ~ 1e-4-1e-5,
    which by itself explains the ~4-order-of-magnitude gain gap observed
    between direct and none in that diagnostic. See module docstring above
    for the w derivation.)

    Chain rule: field = -g, g = grad_z E. For any v,

        J_field^T v = d/dtheta [ v . field ] = d/dtheta [ -v . g ]
                    = - d/dtheta [ v . g ]
                    = - J_g^T v

    and J_g^T v is exactly what the Pearlmutter forward-over-reverse trick
    above computes for JVP direction w=v: dual-forward pass with tangent=v
    gives tangent_E = <g, v> per sample (JVP of E in direction v equals
    <grad_z E, v> by definition of directional derivative), s =
    tangent_E.sum() over the batch, s.backward() accumulates
    d/dtheta[sum_b <g_b, v_b>] = J_g^T v into .grad. This function reuses
    that machinery verbatim with v as the direction (gp_lambda folded out --
    a single free direction, not a residual+regularizer sum), then negates
    the resulting .grad in place to convert J_g^T v -> J_field^T v.

    ACCUMULATES into .grad (does not zero_grad) -- caller must zero_grad()
    immediately before this call for the negation to be well-defined (the
    in-place negate applies to the FULL current .grad, so any grad already
    present before this call would also get incorrectly negated).

    Returns dict: field_norm, v_norm.
    """
    if getattr(model, "ebm", None) != "direct":
        raise ValueError(f"exact_field_vjp requires ebm='direct', got {getattr(model, 'ebm', None)!r}")

    y_dropped, restore_labels = _predrop_labels(model, y)
    restore_attn = _unfused_attention(model)
    try:
      with _forward_ad_safe_ops():
        z = xt.detach().clone().requires_grad_(True)
        E = model(z, t, y_dropped, energy_only=True)
        g = torch.autograd.grad(E.sum(), z, create_graph=False)[0].detach()
        field = -g

        v = v.detach()
        with fwAD.dual_level():
            z_dual = fwAD.make_dual(xt.detach().clone(), v)
            E_dual = model(z_dual, t, y_dropped, energy_only=True)
            _, tangent = fwAD.unpack_dual(E_dual)
            if tangent is None:
                raise RuntimeError(
                    "forward-mode AD produced no tangent through the energy head -- "
                    "an op in the model lacks forward-AD support; do not silently "
                    "fall back (that would reintroduce the semigradient bias)."
                )
            s = tangent.sum()
        s.backward()  # accumulates J_g^T v into p.grad
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.neg_()  # J_g^T v -> J_field^T v = -(J_g^T v)
    finally:
        restore_attn()
        restore_labels()

    return {
        "field_norm": float(field.detach().norm()),
        "v_norm": float(v.detach().norm()),
    }


def field_vjp_none(model, xt, t, y, v):
    """VJP of the field w.r.t. theta for ebm='none' models (field predicted
    directly, no chain rule needed): J_field^T v via ordinary
    torch.autograd.backward(field, grad_tensors=v). Accumulates into .grad
    (caller must zero_grad() first). Companion to exact_field_vjp so both
    arms of matched_replay_jacobian_diagnostic.py compute literally the same
    mathematical object -- a parameter-Jacobian-transpose-vector-product in
    the SAME caller-chosen direction v -- through the same call convention.
    Returns dict: field_norm, v_norm.
    """
    field = model(xt, t, y, train=True)
    v = v.detach()
    torch.autograd.backward(field, grad_tensors=v)
    return {"field_norm": float(field.detach().norm()), "v_norm": float(v.detach().norm())}


# -----------------------------------------------------------------------
# z-space curvature probe (2026-08-10, growing-instability diagnostic)
#
# GP (gradient penalty on mean||grad_z E||^2) provably cannot arrest the
# growing-clip-rate pattern observed in both fwrev arms: quartile forensics
# on jobs 37780076/37780078 showed loss_gp's RELATIVE growth rate is
# identical between the penalized and unpenalized arms (~+0.40%/+0.41% over
# the run) -- GP only rescales the level by a constant ~9% factor, it does
# not touch the slope. A penalty on E[||grad_z E||^2] (a MEAN) has no
# theoretical claim on a TAIL statistic (clip events, driven by rare large
# local curvature) -- Markov/Chebyshev-type bounds require controlling the
# quantity you want a tail guarantee on. The theoretically correct object
# for "is the field about to produce a huge gradient" is the OPERATOR NORM
# of the energy Hessian, ||grad_z^2 E||_op = max_i |eigenvalue_i| -- the
# Lipschitz constant of the field f = -grad_z E. This section estimates it.
# -----------------------------------------------------------------------

def hvp_z(model, z, t, y, v):
    """z-space Hessian-vector product: Hv = (d^2 E / dz^2) v, via standard
    double-backward (create_graph=True on the first grad, ordinary grad for
    the second). OFFLINE DIAGNOSTIC ONLY -- same exemption already used by
    adjoint_optimization.py's jvp_theta_to_cache: this is a frozen-checkpoint
    probe, not a production training step, so the double-backward graph the
    rest of this module exists to avoid (for THETA's gradient) is fine here
    (for Z's Hessian, an offline quantity, never touched by training).

    No TF32/SDPA-backend overrides: deliberately left at whatever the caller
    (train.py, at import time) already configured, so the estimated
    curvature is measured under the SAME numerics that produced the
    grad_norm/clip observations we are trying to explain -- forcing higher
    precision here would decouple the two and invalidate the comparison.

    z: (B, C, H, W), any requires_grad state (cloned+reset internally).
    v: same shape as z -- a batch of probe directions, one per sample.
    Returns Hv, same shape as z. Batched samples are independent (no
    batchnorm / no cross-sample attention in this architecture), so this is
    B independent Hessian-vector products computed in one pass.
    """
    z = z.detach().clone().requires_grad_(True)
    E = model(z, t, y, energy_only=True)
    g = torch.autograd.grad(E.sum(), z, create_graph=True)[0]
    Hv = torch.autograd.grad(g, z, grad_outputs=v, retain_graph=False)[0]
    return Hv.detach()


# -----------------------------------------------------------------------
# theta-space matrix-free JVP + power iteration for sigma_1(J_theta)
# (2026-08-10, longitudinal Jacobian-conditioning diagnostic, Phase 6/7).
#
# The residual-conditioned VJP A(x,r) = ||J_theta^T (r/||r||)|| already
# established (exact_field_vjp/field_vjp_none) answers "how much does the
# ACTUAL residual direction get amplified," but not whether a spike in A
# reflects (a) a genuinely larger top singular value sigma_1(J) at that
# batch ("worse conditioning") or (b) the residual direction u = r/||r||
# happening to align more with an ALREADY-large singular direction
# ("worse alignment") -- since by SVD, A^2 = ||J^T u||^2 =
# sum_i sigma_i^2 (u . u_i)^2 <= sigma_1^2. Distinguishing these requires
# sigma_1 and the top left singular vector u_1 = argmax_i sigma_i's
# direction, independent of whatever u a specific batch happens to produce.
# -----------------------------------------------------------------------

def field_jvp_none(model, xt, t, y, params, v):
    """Matrix-free Jacobian-vector product Jv, J = d(field)/d(params), for
    ebm='none' (field = model(xt,t,y) directly). `params`: an explicit list
    of leaf parameter tensors -- may be a SUBSET of model.parameters()
    (e.g. backbone-only), which restricts J to exactly the columns spanned
    by that subset. `v`: list of tensors matching params' shapes.

    Standard "double-backward for a forward-mode product" (R-operator)
    trick -- avoids needing torch.func.jvp/functorch to differentiate
    w.r.t. an nn.Module's own parameters (fragile: would need in-place
    parameter substitution under a functional wrapper). Two ORDINARY
    reverse passes instead, both first derivatives of new scalars, chained
    via create_graph:

        s(u) = <field(params), u>            u: dummy cotangent, leaf, requires_grad=True
        g(u) = ds/dparams = J^T u            LINEAR in u (grad(..., create_graph=True))
        Jv   = d<g(u), v>/du                  since g(u) = (J^T) u as a function of u,
                                               <g(u), v> = u^T (J v), so its gradient
                                               w.r.t. u IS EXACTLY J v.

    field is differentiated only once (an ordinary autograd graph); the
    "double" backward is w.r.t. (params, holding u fixed) then (u, holding
    the resulting expression fixed) -- no forward-mode AD, but the SECOND
    grad call (Jv = d<g(u),v>/du) IS a genuine double-backward through
    field's own graph, which needs a working double-backward derivative
    for attention; timm's fused F.scaled_dot_product_attention path does
    not provide one ('derivative for ..._flash_attention_..._backward is
    not implemented'). Reuses _unfused_attention (same fix as
    field_jvp_direct above). Validated against an explicitly materialized
    Jacobian on a tiny model in tests/test_fb_direct_exact_hvp.py.
    """
    params = list(params)
    restore_attn = _unfused_attention(model)
    try:
        field = model(xt, t, y, train=True)
        u = torch.zeros_like(field, requires_grad=True)
        s = (field * u).sum()
        # allow_unused=True: a parameter can be mathematically disconnected from
        # field's Jacobian (e.g. an additive-constant-in-z bias term whose
        # contribution to a gradient-of-something-w.r.t.-z vanishes identically)
        # without being a bug -- substitute an exact zero contribution rather than
        # erroring (2026-08-11, found via energy_head.linear.bias in ScalarEnergyHead:
        # E = sum_tokens(W.x_token + bias) = W.sum(x_token) + N_tokens*bias, a
        # z-independent additive constant, so d(field)/d(bias) = 0 exactly).
        if not params:
            raise RuntimeError("field_jvp: `params` is empty -- nothing to differentiate w.r.t.")
        g = torch.autograd.grad(s, params, create_graph=True, allow_unused=True)
        h = None
        for gp, vp in zip(g, v):
            if gp is None:
                continue
            term = (gp * vp).sum()
            h = term if h is None else h + term
        if h is None:
            return torch.zeros_like(field).detach()
        Jv = torch.autograd.grad(h, u)[0]
    finally:
        restore_attn()
    return Jv.detach()


def field_jvp_direct(model, xt, t, y, params, v):
    """Same Jv as field_jvp_none, for ebm='direct' (field = -grad_z E).
    field itself already costs one reverse pass with create_graph=True (E ->
    grad_z E, theta-differentiable); the R-op trick above is layered on top,
    so this is a THIRD-order construction overall -- grad(s, params,
    create_graph=True) is itself a double-backward THROUGH the graph that
    produced gx, i.e. attention needs a working double-backward derivative,
    which timm's fused F.scaled_dot_product_attention path does not provide
    ('derivative for ..._flash_attention_..._backward is not implemented').
    Reuses _unfused_attention (same fix already applied to
    exact_fwrev_backward/exact_field_vjp for a related but distinct reason
    -- there it was forward-AD's in-place-mutation bug; here it is
    double-backward support). Single forward pass (no dual/fwAD machinery,
    unlike exact_field_vjp) -- CFG label dropout is drawn once, no
    cross-pass consistency concern.
    """
    params = list(params)
    restore_attn = _unfused_attention(model)
    try:
        z = xt.detach().clone().requires_grad_(True)
        E = model(z, t, y, energy_only=True)
        gx = torch.autograd.grad(E.sum(), z, create_graph=True)[0]
        field = -gx
        u = torch.zeros_like(field, requires_grad=True)
        s = (field * u).sum()
        if not params:
            raise RuntimeError("field_jvp: `params` is empty -- nothing to differentiate w.r.t.")
        # allow_unused=True: see field_jvp_none's docstring note -- a head parameter
        # can be an additive-constant-in-z term (zero field-Jacobian column) without
        # being a bug.
        g = torch.autograd.grad(s, params, create_graph=True, allow_unused=True)
    finally:
        restore_attn()
    h = None
    for gp, vp in zip(g, v):
        if gp is None:
            continue
        term = (gp * vp).sum()
        h = term if h is None else h + term
    if h is None:
        return torch.zeros_like(field).detach()
    Jv = torch.autograd.grad(h, u)[0]
    return Jv.detach()


def power_iteration_theta_sigma1(jvp_fn, vjp_fn, model, xt, t, y, params, num_iters=20, seed=0, tol=1e-3):
    """Matrix-free power iteration for sigma_1(J) = top singular value of
    J = d(field)/d(params) (params a fixed subset, e.g. backbone-only).

    Alternates v -> q = Jv -> v_new = J^T q (power iteration on the SPD
    operator J^T J), using `jvp_fn`/`vjp_fn` (field_jvp_direct/
    exact_field_vjp or field_jvp_none/field_vjp_none) as the matrix-free
    primitives -- J is never materialized. Tracks sigma_1_est = ||Jv||
    at the CURRENT v each iteration: this equals sigma_1 exactly once v has
    converged to the top right singular vector (standard power-iteration
    Rayleigh estimate), and the corresponding q/||q|| is the top LEFT
    singular vector u_1 -- needed downstream for the alignment diagnostic
    alpha_1 = |<r/||r||, u_1>|.

    Deterministic init (seed) for reproducibility. Early-stops once the
    relative change in sigma_1_est falls below tol (from iteration 2 on),
    else runs num_iters. Validated against numpy SVD of an explicitly
    materialized Jacobian on a tiny model in test_fb_direct_exact_hvp.py.

    vjp_fn is called with the model's FULL parameter set internally
    (matching exact_field_vjp/field_vjp_none's existing signature, which
    always populates .grad for every parameter); this function reads back
    only the `params` subset's .grad afterward, which is exactly
    (J_params)^T q by construction (each parameter's gradient component is
    independent of every other parameter's existence).

    Returns: {"sigma_1": float, "u1": tensor (field shape, unit norm),
    "v1": list of tensors (params shapes, unit norm), "history": list of
    per-iteration sigma_1_est, "n_iters": int}.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    v = [torch.randn(p.shape, generator=gen).to(p.device, p.dtype) for p in params]
    vnorm = sum(float((vp ** 2).sum()) for vp in v) ** 0.5
    v = [vp / vnorm for vp in v]

    history = []
    sigma_prev = None
    q = None
    for _ in range(num_iters):
        model.zero_grad(set_to_none=True)
        q = jvp_fn(model, xt, t, y, params, v)
        sigma_est = float(q.norm())
        history.append(sigma_est)
        if sigma_prev is not None and sigma_prev > 0:
            rel_change = abs(sigma_est - sigma_prev) / sigma_prev
            if rel_change < tol:
                sigma_prev = sigma_est
                break
        sigma_prev = sigma_est

        model.zero_grad(set_to_none=True)
        vjp_fn(model, xt, t, y, q)  # accumulates J_ALL^T q into .grad
        # p.grad may be None for a parameter with a zero field-Jacobian column
        # (e.g. an additive-constant-in-z head bias) -- exact zero, not a bug.
        v_new = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        vnorm_new = sum(float((vp ** 2).sum()) for vp in v_new) ** 0.5
        v = [vp / (vnorm_new + 1e-30) for vp in v_new]
        model.zero_grad(set_to_none=True)

    sigma_1 = sigma_prev
    u1 = (q / (q.norm() + 1e-30)).detach() if q is not None else None
    return {"sigma_1": sigma_1, "u1": u1, "v1": v, "history": history, "n_iters": len(history)}


# -----------------------------------------------------------------------
# Block (orthogonal-iteration) top-k singular SUBSPACE, theta-space.
# (2026-08-10, Stage-A top-k high-gain subspace diagnostic.)
#
# power_iteration_theta_sigma1 above tracks a SINGLE vector; when
# sigma_1 ~ sigma_2 ~ ... the top singular vector is not an individually
# stable object (a small perturbation rotates it inside the near-degenerate
# subspace), even though the SPAN of the top-k subspace is stable. This is
# standard orthogonal iteration (block power method): alternately apply
# J then re-orthonormalize (QR) in output space, apply J^T then
# re-orthonormalize in theta space -- converges the COLUMN SPACE of the
# iterate to the top-k right singular subspace of J, with the QR R-factor
# diagonal converging to the singular values themselves (not merely a
# Rayleigh-quotient proxy), and the output-space orthonormal factor
# converging to the top-k LEFT singular subspace. No deflation, no
# individual-vector bookkeeping -- correct in the presence of
# near-degenerate spectra by construction.
# -----------------------------------------------------------------------

def block_subspace_iteration_theta(jvp_fn, vjp_fn, model, xt, t, y, params, k=8,
                                    num_iters=12, seed=0, tol=1e-3):
    """Top-k singular subspace of J = d(field)/d(params) via orthogonal
    iteration. `params`: fixed subset (e.g. energy-head parameters only).

    Returns:
      sigma: (k,) tensor, descending singular value estimates (final QR
        R-factor diagonal in theta-space, i.e. after the J^T half-step).
      V: (P, k) tensor, orthonormal theta-space basis (flattened param
        order matches `params`), columns = right singular vectors.
      U: (n_out, k) tensor, orthonormal field-space basis (flattened field
        order), columns = left singular vectors, U[:, i] approx J V[:, i] / sigma[i].
      v_list / u_list: same content as V/U but unflattened back to
        param-shape / field-shape tensors, per column, for direct use in
        P_k/Q_k projection metrics.
      history: list of per-iteration sigma vectors (convergence trace).
      ortho_error_V / ortho_error_U: ||V^T V - I||_F / ||U^T U - I||_F at
        the final iterate (should be ~0 by QR construction; reported as an
        explicit numerical sanity check per the spec).
    """
    params = list(params)
    dtype = params[0].dtype
    device = params[0].device
    P = sum(p.numel() for p in params)
    shapes = [p.shape for p in params]

    def unflatten_theta(col):
        out, i = [], 0
        for shp in shapes:
            n = int(torch.tensor(shp).prod()) if len(shp) else 1
            out.append(col[i:i + n].view(shp))
            i += n
        return out

    def flatten_tensors(tensors):
        return torch.cat([t.reshape(-1) for t in tensors])

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    V0 = torch.randn(P, k, generator=gen).to(device=device, dtype=dtype)
    V, _ = torch.linalg.qr(V0)

    field_shape = None
    history = []
    Uq, R_v_diag = None, None
    for it in range(num_iters):
        Q_cols = []
        for j in range(k):
            v_dirs = unflatten_theta(V[:, j])
            model.zero_grad(set_to_none=True)
            Jv = jvp_fn(model, xt, t, y, params, v_dirs)
            if field_shape is None:
                field_shape = Jv.shape
            Q_cols.append(Jv.reshape(-1))
        Q = torch.stack(Q_cols, dim=1)
        Uq, _ = torch.linalg.qr(Q)

        Vnew_cols = []
        for j in range(k):
            u_dir = Uq[:, j].view(field_shape)
            model.zero_grad(set_to_none=True)
            vjp_fn(model, xt, t, y, u_dir)
            # p.grad may be None for a zero-field-Jacobian parameter (see
            # field_jvp_none's docstring note) -- exact zero, not a bug.
            v_new = flatten_tensors([p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                                      for p in params])
            Vnew_cols.append(v_new)
        Vnew = torch.stack(Vnew_cols, dim=1)
        Vnew_q, R_v = torch.linalg.qr(Vnew)
        sigma_est = R_v.diagonal().abs()
        history.append(sigma_est.detach().cpu().tolist())

        if R_v_diag is not None:
            rel_change = float((sigma_est - R_v_diag).abs().max() / (R_v_diag.abs().max() + 1e-30))
            V, R_v_diag = Vnew_q, sigma_est
            if rel_change < tol:
                break
        else:
            V, R_v_diag = Vnew_q, sigma_est
        model.zero_grad(set_to_none=True)

    assert R_v_diag is not None and Uq is not None, "num_iters must be >= 1"
    sigma = R_v_diag
    order = torch.argsort(sigma, descending=True)
    sigma = sigma[order]
    V = V[:, order]
    Uq = Uq[:, order]

    ortho_error_V = float((V.T @ V - torch.eye(V.shape[1], dtype=V.dtype, device=V.device)).norm())
    ortho_error_U = float((Uq.T @ Uq - torch.eye(Uq.shape[1], dtype=Uq.dtype, device=Uq.device)).norm())

    v_list = [unflatten_theta(V[:, j]) for j in range(V.shape[1])]
    u_list = [Uq[:, j].view(field_shape).detach() for j in range(Uq.shape[1])]

    return {
        "sigma": sigma.detach(), "V": V.detach(), "U": Uq.detach(),
        "v_list": v_list, "u_list": u_list, "history": history, "n_iters": len(history),
        "ortho_error_V": ortho_error_V, "ortho_error_U": ortho_error_U,
    }


# -----------------------------------------------------------------------
# Whitened Forward-Backward EqM (WFB-EqM), Stage 0 operator primitives.
# (2026-08-11.) M := d(field)/d(theta) = d^2 E/(dz dtheta), the SAME mixed
# Jacobian already computed by exact_field_vjp (M^T v) and field_jvp_direct
# (M p) above -- no new Jacobian machinery, only the Gram operator
# A = M M^T (field-space -> field-space, PSD symmetric) and a matrix-free
# Lanczos approximation of (A + lambda I)^{-1/2} r built on top of it.
#
# Convention: r = field - ut, UNRESCALED (no 1/(B*D) factor) -- the same
# canonical residual convention as exact_field_vjp's docstring and
# matched_replay_jacobian_diagnostic.py's canonical_residual_and_validate,
# deliberately NOT exact_fwrev_backward's internal w = (2/(B*D))*(...).
# -----------------------------------------------------------------------

def mixed_gram_mv(model, xt, t, y, params, v):
    """A v = M (M^T v), v a field-space tensor (same shape as field/ut).
    `params` must be the SAME full list used consistently across a given
    WFB call (typically list(model.parameters())) -- A's spectrum depends
    on which parameter subset M is restricted to.
    """
    model.zero_grad(set_to_none=True)
    exact_field_vjp(model, xt, t, y, v)  # accumulates M^T v into every p.grad
    Mtv = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    model.zero_grad(set_to_none=True)
    Av = field_jvp_direct(model, xt, t, y, params, Mtv)  # M (M^T v)
    return Av.detach()


def _estimate_lambda_max_generic(gram_mv_fn, v0, num_iters=20, tol=1e-3):
    """Model-agnostic top-eigenvalue power iteration on a PSD operator
    exposed only through a matrix-free `gram_mv_fn(v) -> A v` callable.
    Factored out from estimate_lambda_max so Stage-0 tests can validate the
    pure math against a synthetic operator with a known spectrum, decoupled
    from the real model (same rationale as block_subspace_iteration_theta's
    injectable jvp_fn/vjp_fn above).
    """
    v = v0 / (v0.norm() + 1e-30)
    history = []
    lam_prev = None
    for _ in range(num_iters):
        Av = gram_mv_fn(v)
        lam_est = float((v * Av).sum())  # Rayleigh quotient, ||v||=1
        history.append(lam_est)
        if lam_prev is not None and abs(lam_prev) > 0:
            rel_change = abs(lam_est - lam_prev) / abs(lam_prev)
            if rel_change < tol:
                lam_prev = lam_est
                break
        lam_prev = lam_est
        nrm = float(Av.norm())
        if nrm < 1e-30:
            break
        v = (Av / nrm).detach()
    return {"lambda_max": lam_prev if lam_prev is not None else 0.0,
            "history": history, "n_iters": len(history)}


def estimate_lambda_max(model, xt, t, y, params, num_iters=20, seed=0, tol=1e-3):
    """Matrix-free top-eigenvalue estimate of A = M M^T via ordinary power
    iteration on the field-space Gram operator (mixed_gram_mv). A is PSD by
    construction (it's a Gram matrix), so its dominant eigenvalue equals
    sigma_1(M)^2 and the Rayleigh quotient <v, Av>/<v,v> converges
    monotonically from below.

    Returns {"lambda_max": float, "history": list of per-iter Rayleigh
    estimates, "n_iters": int}.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    # field/residual/target all share xt's shape in this codebase (image tensor
    # in -> field tensor out, same shape) -- no probe call needed.
    v0 = torch.randn(xt.shape, generator=gen).to(xt.device, xt.dtype)
    return _estimate_lambda_max_generic(
        lambda v: mixed_gram_mv(model, xt, t, y, params, v), v0, num_iters=num_iters, tol=tol)


def _lanczos_inv_pow_apply_generic(gram_mv_fn, r, lam, alpha, k=8, reorth=True, eps=1e-10):
    """Model-agnostic matrix-free k-step Lanczos approximation of

        u = (A + lambda I)^{-alpha} r,      A exposed only via gram_mv_fn(v) -> A v.

    Generalizes _lanczos_inv_sqrt_apply_generic (alpha=1/2, WFB) to an
    arbitrary power alpha, giving the family

        g_alpha = M^T (A + lambda I)^{-alpha} r,

    whose induced first-order field update in a singular mode i (M's i-th
    singular value sigma_i) is

        delta_s_i  ~  -eta * sigma_i^2 / (sigma_i^2 + lambda)^alpha * r_i,

    i.e. alpha=0 recovers ordinary direct training (raw M^T r, unbounded
    sigma_i^2 gain), alpha=1/2 is WFB (sigma_i gain -- fixes the PARAMETER
    gradient's conditioning but leaves one power of sigma_i in the induced
    FIELD update), and alpha=1 is full damped Gauss-Newton (FBGN: gain
    sigma_i^2/(sigma_i^2+lambda) in [0,1], i.e. it is the induced field
    update, not just the parameter gradient, that is bounded). See
    documentation/wfb-eqm-stage1-report-2026-08-12.md and the Stage 2.5
    reviewer note (2026-08-12) for the full derivation motivating this
    generalization from the Stage 2 v5/D factorial finding that alpha=1/2
    alone does not permit ordinary-sized optimizer steps.

    Same algorithm as the alpha=1/2 case (Saad 2003 Ch. 13's Lanczos
    approximation to f(A)v, here f(x)=(x+lambda)^{-alpha}), only the final
    eigenvalue transform changes: (T_m+lambda I)^{-alpha} instead of
    (T_m+lambda I)^{-1/2}.

    Builds an orthonormal Krylov basis Q = [q1..qm] (q1 = r/||r||, m<=k) for
    A via three-term recurrence with full reorthogonalization, forms the
    mxm tridiagonal T_m, then uses T_m's tiny explicit eigendecomposition:

        (A + lambda I)^{-alpha} r  ~=  ||r|| * Q  (T_m + lambda I)^{-alpha} e1
                                    =  ||r|| * Q S diag((theta_i+lam)^{-alpha}) S^T e1

    where T_m = S diag(theta) S^T. Standard Lanczos approximation to a
    matrix function applied to a vector (Saad 2003, Ch. 13); T_m's largest
    Ritz value theta_max is also a matrix-free top-eigenvalue estimate of A
    (Kaniel-Paige), returned here as a free diagnostic cross-check against
    estimate_lambda_max's independent power-iteration estimate -- they
    should agree once both have converged.

    Breakdown handling (mandatory, per spec -- never silently substitute
    something else): zero residual, an invariant subspace found early
    (beta_i ~ 0, m < k), or a non-finite A-apply all set breakdown=True with
    an explicit breakdown_reason; callers MUST check this and fail loudly
    rather than proceeding with a degenerate u.

    Factored model-agnostic so Stage-0 tests can validate against an
    explicitly eigendecomposed synthetic operator with a hand-built,
    intentionally extreme singular spectrum, decoupled from the real model
    (same rationale as block_subspace_iteration_theta's injectable-fn
    testing pattern). The model-specific entry point is
    lanczos_inv_sqrt_apply below.

    Returns {"u": tensor (r's shape) or None on breakdown-with-zero-residual,
    "T_eigmax": float, "m": int (Lanczos steps actually taken),
    "breakdown": bool, "breakdown_reason": str or None,
    "ortho_error": float (||Q^T Q - I||_F, numerical sanity check),
    "residual_norm": float}.
    """
    r = r.detach()
    r_norm = float(r.norm())
    if r_norm < eps:
        return {"u": torch.zeros_like(r), "T_eigmax": 0.0, "m": 0,
                "breakdown": True, "breakdown_reason": "zero_residual",
                "ortho_error": 0.0, "residual_norm": r_norm}

    q = r / r_norm
    Q = [q]
    alphas, betas = [], []
    q_prev = torch.zeros_like(r)
    beta_prev = 0.0
    breakdown, breakdown_reason = False, None

    for i in range(k):
        w = gram_mv_fn(Q[-1])
        if not torch.isfinite(w).all():
            breakdown, breakdown_reason = True, f"non_finite_A_apply_at_step_{i}"
            break
        alpha_i = float((w * Q[-1]).sum())
        alphas.append(alpha_i)
        w = w - alpha_i * Q[-1] - beta_prev * q_prev
        if reorth:
            for qj in Q:
                w = w - float((w * qj).sum()) * qj
        if i == k - 1:
            break  # last alpha collected; no need for the (k+1)th basis vector
        beta_i = float(w.norm())
        if beta_i < eps:
            breakdown, breakdown_reason = True, f"invariant_subspace_at_step_{i}"
            break
        betas.append(beta_i)
        q_prev = Q[-1]
        Q.append((w / beta_i).detach())
        beta_prev = beta_i

    m = len(alphas)
    if m == 0:
        return {"u": None, "T_eigmax": 0.0, "m": 0,
                "breakdown": True, "breakdown_reason": breakdown_reason or "no_steps_completed",
                "ortho_error": 0.0, "residual_norm": r_norm}

    # The mxm tridiagonal eigendecomposition is tiny (m<=k, ~O(10)) and computed on CPU
    # regardless of r's device -- but Qmat (the n-dimensional Krylov basis, n = full
    # parameter/field count) lives on r's device, so every CPU-built tensor below must be
    # moved there before combining with Qmat (device mismatch is silent on CPU-only tests,
    # only surfaces on GPU -- found in production, job 38479632).
    device = r.device
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    if m > 1:
        b = torch.tensor(betas, dtype=torch.float64)
        T = T + torch.diag(b, 1) + torch.diag(b, -1)
    theta, S = torch.linalg.eigh(T)  # T = S diag(theta) S^T, ascending theta
    T_eigmax = float(theta.max())

    e1 = torch.zeros(m, dtype=torch.float64)
    e1[0] = 1.0
    coeff = (S @ (torch.diag((theta + lam) ** (-alpha)) @ (S.T @ e1))).to(device)  # (T+lam I)^{-alpha} e1

    Qmat = torch.stack([qi.reshape(-1) for qi in Q], dim=1).to(torch.float64)  # (n, m), on `device`
    ortho_error = float((Qmat.T @ Qmat - torch.eye(m, dtype=torch.float64, device=device)).norm())

    u_flat = r_norm * (Qmat @ coeff.to(Qmat.dtype))
    u = u_flat.view(r.shape).to(r.dtype)

    return {"u": u.detach(), "T_eigmax": T_eigmax, "m": m,
            "breakdown": breakdown, "breakdown_reason": breakdown_reason,
            "ortho_error": ortho_error, "residual_norm": r_norm}


def _lanczos_inv_sqrt_apply_generic(gram_mv_fn, r, lam, k=8, reorth=True, eps=1e-10):
    """alpha=1/2 (WFB) specialization of _lanczos_inv_pow_apply_generic, kept
    as a thin named wrapper for backward compatibility with existing
    Stage 0 callers/tests -- identical math and return contract to before
    this function was generalized, see _lanczos_inv_pow_apply_generic for
    the full docstring/algorithm."""
    return _lanczos_inv_pow_apply_generic(gram_mv_fn, r, lam, alpha=0.5, k=k, reorth=reorth, eps=eps)


def lanczos_inv_sqrt_apply(model, xt, t, y, params, r, lam, k=8, reorth=True, eps=1e-10):
    """Model-specific entry point for _lanczos_inv_sqrt_apply_generic (alpha=1/2,
    WFB), using mixed_gram_mv (A = M M^T for the real model) as the operator.
    See _lanczos_inv_pow_apply_generic's docstring for the algorithm and
    return contract. Thin alpha=0.5 specialization of lanczos_inv_pow_apply
    below, kept for backward compatibility.
    """
    return lanczos_inv_pow_apply(model, xt, t, y, params, r, lam, alpha=0.5, k=k, reorth=reorth, eps=eps)


def lanczos_inv_pow_apply(model, xt, t, y, params, r, lam, alpha, k=8, reorth=True, eps=1e-10):
    """Model-specific entry point for _lanczos_inv_pow_apply_generic, using
    mixed_gram_mv (A = M M^T for the real model) as the operator. See the
    generic function's docstring for the algorithm, the alpha-family
    interpretation (alpha=0 direct, alpha=1/2 WFB, alpha=1 FBGN), and the
    return contract.
    """
    return _lanczos_inv_pow_apply_generic(
        lambda v: mixed_gram_mv(model, xt, t, y, params, v), r, lam, alpha, k=k, reorth=reorth, eps=eps)


def compute_wfb_gradient(model, xt, t, y, ut, params=None, rho=1e-4, k=8,
                          lambda_max_num_iters=20, seed=0, alpha=0.5):
    """Orchestrates the alpha-family Forward-Backward pseudo-gradient

        g_alpha = M^T (A + lambda I)^{-alpha} r,   A = M M^T,
        lambda = rho * lambda_max(A),              r = field - ut  (canonical, unrescaled)

    `alpha` defaults to 0.5 (WFB, the original formulation -- this default
    preserves exact prior behavior for all existing callers). alpha=0 is
    ordinary direct training's raw M^T r (parameter-gradient gain
    sigma_i^2, unbounded); alpha=1 is full damped Gauss-Newton ("FBGN":
    induced FIELD-update gain sigma_i^2/(sigma_i^2+lambda) in [0,1] --
    see _lanczos_inv_pow_apply_generic's docstring for the derivation).
    Despite the name, this function computes g_alpha for whatever alpha is
    passed; the name/docstring/return-key "wfb"/"g_wfb" are kept for
    backward compatibility with existing callers (train.py's
    --wfb-backward, Stage 0-2 tests) rather than renamed wholesale.

    u = (A+lambda I)^{-alpha} r is computed under an implicit stopgrad -- it is
    treated as a fixed direction handed to exact_field_vjp exactly like any
    other caller-supplied v (matching exact_field_vjp's existing contract),
    never differentiated through. field = -grad_z E remains exactly the
    model's native output; only the BACKWARD operator changes. Does not call
    model.zero_grad() on entry/exit beyond what's needed internally -- the
    caller (train.py) owns .grad's final state and must zero_grad() before
    this call, matching exact_fwrev_backward/exact_field_vjp's convention.

    Raises RuntimeError (does not silently degrade) on a genuine numerical
    failure -- a non-finite A-apply, non-finite residual, or zero completed
    Lanczos steps -- per spec, WFB must fail loudly on numerical problems
    rather than falling back to ordinary clipping unannounced. Does NOT
    raise on an "invariant_subspace" Lanczos breakdown (a LUCKY breakdown:
    for the symmetric PSD operator A, this means the Krylov subspace built
    so far is exactly A-invariant, so the reconstruction is EXACT, not
    approximate, in fewer than k steps -- the best possible outcome, not a
    failure) or on a zero residual (nothing to precondition, u=0 is correct).

    Returns dict: g_wfb (list of tensors, matching `params` returned in the
    dict -- see note below), r (tensor), field (tensor), r_norm, lambda_max,
    lam, T_eigmax (Lanczos's own cross-check estimate of lambda_max), m
    (Lanczos steps taken), breakdown, breakdown_reason, ortho_error,
    g_wfb_norm, g_raw (list of tensors -- the hypothetical M^T r, ALWAYS
    computed as a diagnostic per spec Section 9 even when the WFB gradient
    is what gets applied), g_raw_norm, params (the ACTUAL parameter list
    used, after filtering -- see note).

    NOTE: some architectures register fixed (non-trainable) tensors as
    nn.Parameter (e.g. this codebase's sinusoidal `pos_embed`,
    requires_grad=False) so they appear in model.parameters() but are not
    valid autograd.grad `inputs` regardless of allow_unused (that flag only
    covers a requires_grad=True tensor that is disconnected from the graph,
    not a requires_grad=False tensor -- PyTorch raises unconditionally on
    the latter). When `params` is left as the default (None), it is built
    from model.parameters() filtered to requires_grad=True; when the caller
    supplies an explicit `params`, it is used AS GIVEN (caller's
    responsibility) EXCEPT this same filter is still applied, since silently
    including a requires_grad=False tensor would crash rather than degrade
    gracefully, and there is no meaningful WFB contribution from a frozen
    parameter (its true Jacobian column is simply never trained). Callers
    that need to align a params-shaped list elsewhere (e.g. splitting
    g_wfb by architecture group) MUST use the returned `params`, not their
    own unfiltered list, to avoid a length/order mismatch.
    """
    params_in = list(params) if params is not None else list(model.parameters())
    params = [p for p in params_in if p.requires_grad]
    if len(params) != len(params_in):
        dropped = len(params_in) - len(params)
        print(f"[compute_wfb_gradient] filtered {dropped} requires_grad=False parameter(s) out of "
              f"the {len(params_in)}-tensor params list (e.g. a fixed/frozen buffer registered as "
              f"nn.Parameter) -- WFB only operates on trainable parameters.")
    if not params:
        raise RuntimeError("compute_wfb_gradient: no trainable (requires_grad=True) parameters in `params`.")

    # Predraw CFG label dropout ONCE and hold it fixed for every internal model call below
    # (compute_field_direct, exact_field_vjp x{1 + Lanczos steps}, field_jvp_direct x
    # {lambda_max power-iters + Lanczos steps}) -- same fix, same reason, as
    # exact_fwrev_backward/exact_field_vjp's existing _predrop_labels usage: this function
    # calls the model MANY times to build A = M M^T and apply (A+lambda I)^{-1/2}, and if
    # model.training=True with dropout_prob>0 (true during actual training, never true in
    # Stage 0/1's model.eval()-only usage), an independent dropout draw per call would
    # silently make M a DIFFERENT operator on every application -- violating the fixed-
    # linear-operator assumption the whole Lanczos three-term recurrence depends on, which
    # manifests as numerical garbage (confirmed in production, WFB-EqM Stage 2 job
    # 38493610: 'Lanczos produced a non-finite/absent u' on literally the first training
    # step, the first time this function ran with model.training=True).
    y, restore_labels = _predrop_labels(model, y)
    try:
        model.zero_grad(set_to_none=True)
        field = compute_field_direct(model, xt, t, y)
        r = (field - ut).detach()
        r_norm = float(r.norm())
        if not torch.isfinite(r).all():
            raise RuntimeError("compute_wfb_gradient: residual r is non-finite -- failing loudly, not degrading to a fallback.")

        # Hypothetical raw gradient M^T r -- diagnostic, always computed (Section 9/18).
        model.zero_grad(set_to_none=True)
        exact_field_vjp(model, xt, t, y, r)
        g_raw = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        g_raw_norm = sum(float((gp ** 2).sum()) for gp in g_raw) ** 0.5

        lam_result = estimate_lambda_max(model, xt, t, y, params, num_iters=lambda_max_num_iters, seed=seed)
        lambda_max = lam_result["lambda_max"]
        lam = rho * lambda_max

        lz = lanczos_inv_pow_apply(model, xt, t, y, params, r, lam, alpha, k=k)
        reason = lz["breakdown_reason"]
        # "invariant_subspace_at_step_i" is a LUCKY Lanczos breakdown, not a failure: for
        # symmetric operators (A is PSD-symmetric by construction) beta_i=0 means the Krylov
        # subspace built so far is exactly A-invariant, so the m-term tridiagonal
        # reconstruction of (A+lam I)^{-1/2} r is EXACT (not approximate) restricted to that
        # subspace -- and since r itself lies in it, this IS the exact global answer, achieved
        # in fewer than k steps. "zero_residual" (nothing to precondition) is also benign.
        # Only a genuine numerical failure (non-finite A-apply, or zero completed steps) fails loudly.
        fatal = lz["breakdown"] and (reason == "no_steps_completed" or (reason or "").startswith("non_finite"))
        if fatal:
            raise RuntimeError(f"compute_wfb_gradient: Lanczos breakdown ({reason}) at "
                                f"m={lz['m']}/{k} -- failing loudly per spec, not silently substituting.")
        u = lz["u"]
        if u is None or not torch.isfinite(u).all():
            raise RuntimeError("compute_wfb_gradient: Lanczos produced a non-finite/absent u.")

        model.zero_grad(set_to_none=True)
        exact_field_vjp(model, xt, t, y, u)  # accumulates g_wfb = M^T u into .grad
        g_wfb = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        g_wfb_norm = sum(float((gp ** 2).sum()) for gp in g_wfb) ** 0.5
    finally:
        restore_labels()

    return {
        "params": params,
        "g_wfb": g_wfb, "g_wfb_norm": g_wfb_norm,
        "g_raw": g_raw, "g_raw_norm": g_raw_norm,
        "r": r, "r_norm": r_norm, "field": field.detach(),
        "lambda_max": lambda_max, "lambda_max_history": lam_result["history"], "lam": lam,
        "alpha": alpha,
        "T_eigmax": lz["T_eigmax"], "m": lz["m"],
        "breakdown": lz["breakdown"], "breakdown_reason": lz["breakdown_reason"],
        "ortho_error": lz["ortho_error"],
    }


def compute_field_direct(model, xt, t, y):
    """field = -grad_z E_theta(z), the model's native ebm='direct' output,
    computed WITHOUT create_graph (no theta-graph retained) -- used wherever
    WFB needs the field value itself as a diagnostic, independent of any
    particular VJP/JVP direction.
    """
    if getattr(model, "ebm", None) != "direct":
        raise ValueError(f"compute_field_direct requires ebm='direct', got {getattr(model, 'ebm', None)!r}")
    restore_attn = _unfused_attention(model)
    y_dropped, restore_labels = _predrop_labels(model, y)
    try:
        z = xt.detach().clone().requires_grad_(True)
        E = model(z, t, y_dropped, energy_only=True)
        g = torch.autograd.grad(E.sum(), z, create_graph=False)[0]
    finally:
        restore_attn()
        restore_labels()
    return (-g).detach()


def power_iteration_spectral_norm(model, z, t, y, num_iters=15, seed=None):
    """Per-sample power iteration for ||grad_z^2 E||_op = max_i |eigenvalue_i|
    (spectral RADIUS, not just the top positive eigenvalue -- E is generally
    non-convex so the Hessian is indefinite; unshifted power iteration on a
    symmetric matrix converges to the eigenvector of largest |eigenvalue|
    regardless of its sign, oscillating u's sign each step if that
    eigenvalue is negative -- the Rayleigh quotient's ABSOLUTE VALUE still
    converges monotonically, which is what we track and return. Verified
    against a full eigh() decomposition on a tiny model in
    tests/test_fb_direct_curvature_probe.py).

    Runs power iteration independently PER SAMPLE (normalizing each
    sample's probe vector by its own norm each iteration), not globally
    across the batch -- correct because this architecture's Hessian of
    E.sum() w.r.t. the batched z is block-diagonal across samples (no
    batchnorm, self-attention never mixes across the batch dimension).

    Returns: {"spectral_norm": (B,) tensor of |top eigenvalue| estimates,
              "rayleigh_history": list of (B,) tensors, one per iteration,
              for convergence inspection}.
    """
    B = z.shape[0]
    gen = torch.Generator(device="cpu") if seed is not None else None
    if gen is not None:
        gen.manual_seed(seed)
    u = torch.randn(z.shape, generator=gen).to(z.device, z.dtype)
    u = u / u.flatten(1).norm(dim=1).clamp_min(1e-30).view(-1, *([1] * (u.dim() - 1)))

    history = []
    rq = None
    for _ in range(num_iters):
        Hu = hvp_z(model, z, t, y, u)
        rq = (u * Hu).flatten(1).sum(dim=1).abs()  # |Rayleigh quotient| per sample
        history.append(rq.detach().cpu())
        norm = Hu.flatten(1).norm(dim=1).clamp_min(1e-30)
        u = (Hu / norm.view(-1, *([1] * (Hu.dim() - 1)))).detach()

    return {"spectral_norm": rq.detach(), "rayleigh_history": history}
