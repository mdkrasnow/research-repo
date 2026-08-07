"""
Cache-adjoint decomposition: the mandatory pre-registered numerical check
(Section 5 of the learned-cache-adjoint proposal, 2026-08-07) that must pass
before any corrector network is trained.

Exact parameter gradient decomposes as

    g_exact = g_semi + g_cache,        g_cache := g_exact - g_semi

(see fb_direct/trainer.py's training_step docstring). g_exact and g_semi are
each independently well-defined (real double-backward through theta's
unmodified forward pass; and today's production semigradient step,
respectively) -- g_cache is DEFINED as their difference, not measured
directly. This module tests whether

    g_cache =?= J_{C_theta}(theta)^T a*,   a* := dL_fb/dC

reconstructs that difference via an ordinary (first-order) VJP, where C is
the forward cache (attention q/k/v/p, MLP pre-activation, LayerNorm
normalized/inv_std per block, conditioning c, final-layer LayerNorm state).

TWO EASY-TO-GET-WRONG POINTS, both caught by this module's own earlier
failing drafts (see `documentation/fb_direct_cache_adjoint_decomposition.md`):

1. WHICH a*.  a* is dL_fb/dC computed through phi's OWN manual reverse-VJP
   formulas (R_phi) -- NOT dL/dC computed via theta's real downstream
   forward computation (patch/blocks/energy_head continuing normally past
   C). Those are different quantities: the latter mixes in theta's role as
   R's own "coefficients", which g_semi already accounts for.

2. CACHE TENSORS MUST BE MUTUALLY INDEPENDENT LEAVES when computing a*.
   In PRODUCTION, forward_cache.forward_energy_with_cache runs under
   `torch.no_grad()` and phi consumes the resulting numbers as independent
   inputs -- there is no graph edge from `normalized1` to `q` in phi's
   world, even though theta's OWN forward pass computed q FROM normalized1.
   If a* is instead extracted via retain_grad() on cache tensors that are
   STILL graph-connected to each other (e.g. by simply not detaching
   forward_energy_with_cache_grad's output), an earlier tensor's captured
   gradient silently DOUBLE-COUNTS the downstream tensors' consumption of
   it (q's phi-driven gradient spuriously backflows through normalized1,
   even though phi itself never derives q from normalized1). This module's
   first attempt at this test hit exactly that bug: cosine(g_cache_vjp,
   g_cache_direct) topped out at ~0.93, not >0.999, purely from this
   graph-cutting error -- not a finding about the underlying hypothesis.

   Fix: obtain the NUMERIC cache via the production (no_grad) path, then
   re-wrap each tensor as an INDEPENDENT leaf with requires_grad_(True)
   before feeding it to phi. No cross-tensor graph edges exist by
   construction, so a* is exactly phi's local per-tensor sensitivity.

Given (1) and (2), J_{C_theta}(theta)^T a* is computed via a SEPARATE,
theta-only forward pass (forward_energy_with_cache_grad, which keeps the
running hidden state `x` genuinely differentiable across ALL blocks --
that cross-block connectivity is exactly what makes g_cache "distributed
through depth" per the layerwise truncation finding, job 37541376) using
`torch.autograd.grad(cache_tensors, theta_params, grad_outputs=a_star)`, an
ordinary first-order VJP with fixed (detached) cotangents.
"""
import torch

from .forward_cache_grad import forward_energy_with_cache_grad


def mean_flat(x):
    return x.mean(dim=list(range(1, len(x.shape))))


def _tf32_disabled():
    class _Guard:
        def __enter__(self):
            self.cuda_prev = torch.backends.cuda.matmul.allow_tf32
            self.cudnn_prev = torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            torch.backends.cuda.matmul.allow_tf32 = self.cuda_prev
            torch.backends.cudnn.allow_tf32 = self.cudnn_prev
            return False
    return _Guard()


def _leaf_copy_cache(cache):
    """Re-wrap every tensor in a (detached, numerically frozen) production
    TransformerReverseCache as an INDEPENDENT leaf with requires_grad=True,
    so phi's consumption of one tensor cannot leak into another's gradient
    via any shared graph history (there is none -- these are fresh leaves).
    Returns (leaf_cache, {name: leaf_tensor}) -- leaf_cache has the exact
    same nested dataclass shape phi.forward() expects.
    """
    import copy
    leaf_cache = copy.copy(cache)
    leaf_cache.blocks = []
    leaves = {}

    def leafify(t, name):
        leaf = t.detach().clone().requires_grad_(True)
        leaves[name] = leaf
        return leaf

    from .cache import (
        AttentionReverseCache, FinalLayerReverseCache, LayerNormReverseCache,
        MLPReverseCache, SiTBlockReverseCache,
    )

    leaf_cache.c = leafify(cache.c, "c")
    for i, b in enumerate(cache.blocks):
        leaf_cache.blocks.append(SiTBlockReverseCache(
            norm1=LayerNormReverseCache(
                normalized=leafify(b.norm1.normalized, f"blocks.{i}.norm1.normalized"),
                inv_std=leafify(b.norm1.inv_std, f"blocks.{i}.norm1.inv_std"),
            ),
            attn=AttentionReverseCache(
                q=leafify(b.attn.q, f"blocks.{i}.attn.q"),
                k=leafify(b.attn.k, f"blocks.{i}.attn.k"),
                v=leafify(b.attn.v, f"blocks.{i}.attn.v"),
                p=leafify(b.attn.p, f"blocks.{i}.attn.p"),
            ),
            norm2=LayerNormReverseCache(
                normalized=leafify(b.norm2.normalized, f"blocks.{i}.norm2.normalized"),
                inv_std=leafify(b.norm2.inv_std, f"blocks.{i}.norm2.inv_std"),
            ),
            mlp=MLPReverseCache(pre_act=leafify(b.mlp.pre_act, f"blocks.{i}.mlp.pre_act")),
        ))
    leaf_cache.final = FinalLayerReverseCache(
        norm_final=LayerNormReverseCache(
            normalized=leafify(cache.final.norm_final.normalized, "final.norm_final.normalized"),
            inv_std=leafify(cache.final.norm_final.inv_std, "final.norm_final.inv_std"),
        )
    )
    return leaf_cache, leaves


def compute_g_exact(theta, xt, t, y, ut):
    """The TRUE gradient: real double-backward through theta's own,
    UNMODIFIED forward computation. Uses forward_energy_with_cache_grad
    purely as a numerically-identical stand-in for theta's ordinary forward
    pass (verified in tests/test_fb_direct_cache_adjoint.py) so xt's
    requires_grad plumbing is uniform with the rest of this module; this
    function never reads or writes any cache tensor's .grad, only theta's
    own parameters'.
    """
    with _tf32_disabled():
        z_req = xt.detach().clone().requires_grad_(True)
        E, _ = forward_energy_with_cache_grad(theta, z_req, t, y)
        grad_z = torch.autograd.grad(E.sum(), z_req, create_graph=True)[0]
        field = -grad_z
        loss = mean_flat((field - ut) ** 2).mean()

        theta.zero_grad(set_to_none=True)
        loss.backward()
        g_exact = {
            name: p.grad.detach().reshape(-1).clone()
            for name, p in theta.named_parameters()
            if p.grad is not None
        }
        loss_exact = float(loss.detach())
    theta.zero_grad(set_to_none=True)
    return g_exact, loss_exact


def compute_g_semi_and_a_star(fb_trainer, xt, t, y, ut):
    """Production-equivalent semigradient step, EXCEPT the numeric cache is
    obtained via `forward_energy_with_cache_grad` under `torch.no_grad()`
    (the SAME forward code `compute_g_cache_vjp` uses to build its graph)
    rather than `forward_cache.forward_energy_with_cache`'s separate
    implementation.

    Retune (2026-08-07, after job 37676460 found median cosine 0.940 on the
    real checkpoint, short of the mandatory >0.999): the two forward
    functions are unit-tested to produce numerically IDENTICAL energy
    (test_forward_cache_matches_direct_energy /
    test_forward_cache_grad_matches_direct_energy), but "identical to
    float64 test tolerance" is not "bit-identical at FP32 on a 12-block
    A100 forward pass" -- any operation-ordering difference between the two
    implementations would compound through the same TF32/FP32-accumulation
    depth-compounding mechanism Gate 0 already diagnosed (job 37520759:
    ~1.5e-4 relative field error from this exact mechanism), except here
    a* feeds a SECOND differentiation (the VJP reconstruction), which is
    more sensitive to input noise than a single forward evaluation. Using
    the identical code path for both a*'s source cache and the VJP
    reconstruction's cache eliminates that as a candidate divergence
    source entirely (any remaining gap is then unambiguously about the
    THEORY, not implementation drift).

    Each cache tensor is still re-wrapped as an independent
    `requires_grad=True` leaf before being handed to phi -- see
    `_leaf_copy_cache`'s docstring for why this independence is mandatory
    (identical dataclass field layout between TransformerReverseCache and
    TransformerGradCache, so `_leaf_copy_cache` works unchanged on either).
    phi.grad here is bit-identical to production's g_semi (detaching cache
    never changes phi's own gradient); the only new thing extracted is
    a* = each leaf's post-backward .grad.
    """
    theta = fb_trainer.theta
    phi = fb_trainer.phi
    fb_trainer.registry.tie_from_forward_()
    for p in phi.parameters():
        p.grad = None

    with _tf32_disabled():
        with torch.no_grad():
            z0 = xt.detach().clone()
            _, cache = forward_energy_with_cache_grad(theta, z0, t, y)
        leaf_cache, leaves = _leaf_copy_cache(cache)

        grad_tilde = phi(leaf_cache)
        prediction_fb = -grad_tilde
        loss_fb = mean_flat((prediction_fb - ut) ** 2).mean()
        loss_fb.backward()

        a_star = {
            name: leaf.grad.detach().clone()
            for name, leaf in leaves.items()
            if leaf.grad is not None
        }
        phi_named = dict(phi.named_parameters())
        g_semi = {}
        for e in fb_trainer.registry.entries:
            if e.category not in ("reverse_active", "recomputed_conditioning"):
                continue
            p = phi_named[e.backward_name]
            if p.grad is not None:
                g_semi[e.forward_name] = p.grad.detach().reshape(-1).clone()
        loss_fb_val = float(loss_fb.detach())

    for p in phi.parameters():
        p.grad = None
    return g_semi, a_star, loss_fb_val, cache


def compute_g_cache_vjp(theta, xt, t, y, a_star, compute_per_tensor_contribution=True):
    """Ordinary (first-order) adjoint-replay VJP g_cache_vjp = J_C^T
    stopgrad(a*): a FRESH forward pass building the graph theta -> cache
    tensors (running hidden state `x` fully differentiable across ALL
    blocks, so each cache tensor's genuine cascading dependence on EARLIER
    blocks' theta is captured -- this is what makes g_cache "distributed
    through depth"), then a single
    `torch.autograd.grad(cache_tensors, theta_params, grad_outputs=a_star)`.
    No second-order derivative anywhere in this function -- this is exactly
    the operation a trained corrector's (detached) output would be fed
    through at inference/production time.

    Returns:
      g_cache_vjp: {theta_param_name: detached flat grad tensor}
      per_tensor_contribution: {cache_tensor_name: float}, the L2 norm of
        that single tensor's contribution to g_cache_vjp (via a fresh
        single-tensor VJP), for the cache tensor inventory report. Costs
        ~1 extra forward+backward pass PER cache tensor (~100+ for a
        depth-12 model) -- set compute_per_tensor_contribution=False on
        most batches (this is diagnostic/inventory info, not part of the
        mandatory cosine gate) or this dominates runtime.
    """
    with _tf32_disabled():
        z2 = xt.detach().clone()
        _, cache2 = forward_energy_with_cache_grad(theta, z2, t, y)
        cache2_flat = dict(cache2.flatten())

        # Cache tensors produced solely by cache_only (frozen) modules --
        # e.g. `c` via t_embedder/y_embedder -- carry no theta-VJP signal at
        # all once ForwardBackwardsDirectTrainer.__init__ has frozen those
        # parameters (param_mapping.py CATEGORY_CACHE_ONLY); their true
        # contribution to g_cache is exactly zero, so skip them here rather
        # than erroring on a tensor with no grad_fn.
        names = [n for n in a_star if n in cache2_flat and cache2_flat[n].requires_grad]
        tensors = [cache2_flat[n] for n in names]
        grad_outputs = [a_star[n].to(cache2_flat[n].dtype) for n in names]

        theta_params = [p for p in theta.parameters() if p.requires_grad]
        theta_names = [n for n, p in theta.named_parameters() if p.requires_grad]

        grads = torch.autograd.grad(
            tensors, theta_params, grad_outputs=grad_outputs,
            retain_graph=False, allow_unused=True,
        )
        g_cache_vjp = {
            name: (g.detach().reshape(-1).clone() if g is not None
                   else torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            for name, p, g in zip(theta_names, theta_params, grads)
        }

    # Per-cache-tensor contribution: a SEPARATE single-tensor VJP for each
    # name, so callers can rank which cache families actually move g_cache
    # (Section 6's inventory). This re-forwards once per tensor -- fine for
    # a diagnostic script, not used on any training path.
    per_tensor_contribution = {}
    if not compute_per_tensor_contribution:
        return g_cache_vjp, per_tensor_contribution
    with _tf32_disabled():
        for n in names:
            z3 = xt.detach().clone()
            _, cache3 = forward_energy_with_cache_grad(theta, z3, t, y)
            cache3_flat = dict(cache3.flatten())
            tensor = cache3_flat[n]
            go = a_star[n].to(tensor.dtype)
            theta_params_n = [p for p in theta.parameters() if p.requires_grad]
            grads_n = torch.autograd.grad(
                [tensor], theta_params_n, grad_outputs=[go],
                retain_graph=False, allow_unused=True,
            )
            flat = torch.cat([
                g.detach().reshape(-1).float() if g is not None
                else torch.zeros(p.numel(), device=p.device)
                for g, p in zip(grads_n, theta_params_n)
            ])
            per_tensor_contribution[n] = float(flat.norm())

    return g_cache_vjp, per_tensor_contribution


def decomposition_test(fb_trainer, active_pairs, xt, t, y, ut, compute_per_tensor_contribution=True):
    """Runs the full mandatory decomposition test on one batch.

    compute_per_tensor_contribution: pass False on most batches -- it costs
      ~1 extra forward+backward pass PER cache tensor family and is purely
      Section-6 inventory info, not part of the mandatory cosine gate.

    Returns a dict with:
      cosine_g_cache_vjp_vs_direct, rel_norm_error_g_cache (the mandatory
        Section-5 numerical check: does J_C^T a* reconstruct g_exact-g_semi),
      cosine_g_semi_vs_exact (baseline, should match Gate 1's 0.62-0.66 on
        the real checkpoint),
      loss_exact, loss_fb, per_tensor_contribution, a_star_rms,
      cache_tensor_shapes (for the Section 6 inventory report).
    """
    theta = fb_trainer.theta
    active_names = {fn for fn, _ in active_pairs}

    g_exact, loss_exact = compute_g_exact(theta, xt, t, y, ut)
    g_semi, a_star, loss_fb, _cache = compute_g_semi_and_a_star(
        fb_trainer, xt, t, y, ut
    )
    g_cache_vjp, per_tensor_contribution = compute_g_cache_vjp(
        theta, xt, t, y, a_star, compute_per_tensor_contribution=compute_per_tensor_contribution
    )

    exact_parts, semi_parts, vjp_parts = [], [], []
    for fn in sorted(active_names):
        if fn not in g_exact or fn not in g_semi or fn not in g_cache_vjp:
            continue
        exact_parts.append(g_exact[fn])
        semi_parts.append(g_semi[fn])
        vjp_parts.append(g_cache_vjp[fn])
    exact_vec = torch.cat(exact_parts)
    semi_vec = torch.cat(semi_parts)
    vjp_vec = torch.cat(vjp_parts)

    g_cache_direct = exact_vec - semi_vec
    g_hat = semi_vec + vjp_vec

    def cosine(a, b):
        return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    def rel_norm_err(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    cache_tensor_shapes = {name: tuple(tensor.shape) for name, tensor in a_star.items()}
    a_star_rms = {name: float(a_star[name].float().pow(2).mean().sqrt()) for name in a_star}

    return {
        "cosine_g_cache_vjp_vs_direct": cosine(vjp_vec, g_cache_direct),
        "rel_norm_error_g_cache": rel_norm_err(vjp_vec, g_cache_direct),
        "cosine_g_hat_vs_exact": cosine(g_hat, exact_vec),
        "cosine_g_semi_vs_exact": cosine(semi_vec, exact_vec),
        "norm_g_cache_direct": float(g_cache_direct.norm()),
        "norm_g_cache_vjp": float(vjp_vec.norm()),
        "norm_g_exact": float(exact_vec.norm()),
        "norm_g_semi": float(semi_vec.norm()),
        "loss_exact": loss_exact,
        "loss_fb": loss_fb,
        "per_tensor_contribution": per_tensor_contribution,
        "a_star_rms": a_star_rms,
        "cache_tensor_shapes": cache_tensor_shapes,
    }
