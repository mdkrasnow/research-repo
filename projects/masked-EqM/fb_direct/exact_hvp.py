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
        g = torch.autograd.grad(s, params, create_graph=True)
        if not g:
            raise RuntimeError("field_jvp: `params` is empty -- nothing to differentiate w.r.t.")
        h = None
        for gp, vp in zip(g, v):
            term = (gp * vp).sum()
            h = term if h is None else h + term
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
        g = torch.autograd.grad(s, params, create_graph=True)
    finally:
        restore_attn()
    if not g:
        raise RuntimeError("field_jvp: `params` is empty -- nothing to differentiate w.r.t.")
    h = None
    for gp, vp in zip(g, v):
        term = (gp * vp).sum()
        h = term if h is None else h + term
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
        v_new = [p.grad.detach().clone() for p in params]
        vnorm_new = sum(float((vp ** 2).sum()) for vp in v_new) ** 0.5
        v = [vp / (vnorm_new + 1e-30) for vp in v_new]
        model.zero_grad(set_to_none=True)

    sigma_1 = sigma_prev
    u1 = (q / (q.norm() + 1e-30)).detach() if q is not None else None
    return {"sigma_1": sigma_1, "u1": u1, "v1": v, "history": history, "n_iters": len(history)}


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
