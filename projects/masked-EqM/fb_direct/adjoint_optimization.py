"""
Test A of Yilun's post-decomposition decision tree (2026-08-07): the TRUE
representational ceiling of the cache-adjoint correction, as opposed to the
oracle-a* performance measured by fb_direct/cache_adjoint.py.

Given b := g_cache_direct = g_exact - g_semi (fixed, theta-space), a* (the
adjoint of phi's OWN reverse formulas w.r.t. their cache input) achieves
only cosine(J_C^T a*, b) ~ 0.94 on the real epoch-40 checkpoint
(job 37676460/37684746). Because phi only approximately reproduces the true
local reverse-mode Jacobian (Gate 0: ~1.5e-4 relative field error, never
exact), a* is the adjoint appropriate to an IMPERFECT operator R_phi, not
necessarily the best possible adjoint for reconstructing b through the
TRUE linear map J_C(theta): theta -> cache (fixed theta, this checkpoint).

This module solves, independently of phi and a* entirely,

    a_best = argmin_a || J_C(theta)^T a - b ||^2

via CG on the normal equations (CGNR), using:
  - VJP  A v  := J_C^T v   (cache-space -> theta-space): `vjp_cache_to_theta`
  - JVP  A^T u := J_C u    (theta-space -> cache-space): `jvp_theta_to_cache`,
    via the standard double-backward trick (dummy dual variables). This is a
    genuine double-backward, which is fine here -- Test A is an OFFLINE
    diagnostic on a frozen checkpoint, not a production training step; it
    never touches fb_direct/trainer.py's training_step.

Interpretation (per Yilun's decision tree):
  rho_best ~ 0.94   -> the ~0.94 ceiling is structural: a* is already
                       close to optimal given this decomposition, so
                       cheap non-representational reasons don't explain
                       the gap. Move to Test B (oracle-a* continuation).
  rho_best >> 0.94   -> a* is a suboptimal adjoint for the (approximate)
                       operator J_C; a LEARNED, deliberately-biased
                       corrector could beat oracle-a* performance. This
                       reframes what should be learned: not a*, but
                       whatever a_best is approximating.
"""
import gc

import torch
from torch.func import jvp as torch_func_jvp
from torch.nn.utils.stateless import _reparametrize_module

from .forward_cache_grad import forward_energy_with_cache_grad


def _release_autograd_cycles():
    """create_graph=True double-backward graphs (used by jvp_theta_to_cache)
    routinely contain reference cycles (grad_fn <-> saved tensors) that
    plain refcounting cannot free -- only the cyclic garbage collector can.
    Without this, genuinely-allocated (not just fragmented/reserved) CUDA
    memory grows monotonically across CGNR iterations until OOM: job
    37688514 hit this on a FULL 80GB A100 (75.8GB allocated) crashing on
    the very first batch's CGNR loop, at the same call site every time,
    which rules out a single-call sizing problem and points at exactly
    this well-known create_graph=True gotcha.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _dict_dot(a, b):
    total = None
    for k, va in a.items():
        vb = b.get(k)
        if vb is None:
            continue
        d = (va.reshape(-1).double() * vb.reshape(-1).double()).sum()
        total = d if total is None else total + d
    return float(total) if total is not None else 0.0


def _active_theta_params(theta):
    """(names, params) for exactly the theta parameters with requires_grad=True
    -- by construction (param_mapping.py's freeze_cache_only_()) this is
    precisely the reverse_active + recomputed_conditioning set, i.e. the
    same space g_exact/g_semi/g_cache_direct all live in.
    """
    names, params = [], []
    for n, p in theta.named_parameters():
        if p.requires_grad:
            names.append(n)
            params.append(p)
    return names, params


def vjp_cache_to_theta(theta, xt, t, y, a):
    """A v := J_C(theta)^T a, an ordinary (first-order) VJP: fresh forward
    pass building the graph theta -> cache tensors, then
    torch.autograd.grad(cache_tensors, theta_params, grad_outputs=a).
    Mechanically identical to fb_direct/cache_adjoint.py's
    compute_g_cache_vjp, but parametrized by an arbitrary cache-space
    dict `a` instead of hardcoding a_star -- kept as a separate
    implementation here (not a shared import) so Test A cannot regress the
    already-verified Section-5 gate code.

    Returns: {theta_param_name: flat tensor}, native dtype.
    """
    with _tf32_disabled():
        z = xt.detach().clone()
        _, cache = forward_energy_with_cache_grad(theta, z, t, y)
        cache_flat = dict(cache.flatten())

        names = [n for n in a if n in cache_flat and cache_flat[n].requires_grad]
        tensors = [cache_flat[n] for n in names]
        grad_outputs = [a[n].to(cache_flat[n].dtype) for n in names]

        theta_names, theta_params = _active_theta_params(theta)
        grads = torch.autograd.grad(
            tensors, theta_params, grad_outputs=grad_outputs,
            retain_graph=False, allow_unused=True,
        )
        return {
            name: (g.detach().reshape(-1).clone() if g is not None
                   else torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            for name, p, g in zip(theta_names, theta_params, grads)
        }


def jvp_theta_to_cache(theta, xt, t, y, direction):
    """A^T u := J_C(theta) direction, computed via `torch.func.jvp` -- TRUE
    forward-mode automatic differentiation (dual numbers), not a manual
    double-backward trick.

    Replaces this module's original hand-rolled implementation (dummy dual
    variables + create_graph=True + a second autograd.grad call), which
    exhibited a genuine, reproducible ~3.7GB-per-call CUDA memory leak on
    the real checkpoint (jobs 37688514/37689193, isolated and confirmed by
    fb_direct_testA_memory_probe.py/job 37689738) that neither `del` nor
    explicit gc.collect()+torch.cuda.empty_cache() could reclaim -- the
    leak's root cause in the manual version was never fully identified.
    Forward-mode AD never builds or retains a backward graph at all (it
    propagates tangents alongside the primal computation in a single
    forward pass), so there is no analogous per-call state to leak, and it
    is the officially maintained implementation of exactly this operation.

    Uses `torch.nn.utils.stateless._reparametrize_module` -- the same
    mechanism `torch.func.functional_call` uses internally -- to run
    `forward_energy_with_cache_grad` (an external function that reads
    theta's submodules directly, not `theta.forward()`) as a pure function
    of the given parameter tensors; `functional_call` itself can only call
    a module's own `forward()`, so the lower-level primitive is required
    here. This is the standard, documented pattern for combining
    `torch.func` transforms with an nn.Module's parameters.

    Cache tensors produced solely by frozen (cache_only) modules -- e.g.
    `c` via t_embedder/y_embedder -- need NO special-casing here (unlike
    the old VJP-based implementation): forward-mode AD naturally propagates
    a zero tangent through any computation that never touches a
    differentiated primal, since those modules' parameters are simply held
    constant (not part of `theta_names`/`primals`) during tracing.

    direction: {theta_param_name: tensor}, missing entries treated as zero.
    Returns: {cache_tensor_name: tensor at its natural shape}, native dtype.
    """
    theta_names, theta_params = _active_theta_params(theta)
    primals = tuple(p.detach() for p in theta_params)
    tangents = tuple(
        direction.get(n, torch.zeros(p.numel(), device=p.device, dtype=p.dtype)).reshape(p.shape)
        for n, p in zip(theta_names, theta_params)
    )
    z = xt.detach().clone()
    names_box = {}

    def cache_fn(*param_values):
        params = dict(zip(theta_names, param_values))
        with _reparametrize_module(theta, params), _tf32_disabled():
            _, cache = forward_energy_with_cache_grad(theta, z, t, y)
        cache_items = cache.flatten()
        names_box["names"] = [n for n, _ in cache_items]
        return tuple(c for _, c in cache_items)

    _, tangent_out = torch_func_jvp(cache_fn, primals, tangents)
    return {name: g.detach().clone() for name, g in zip(names_box["names"], tangent_out)}


def cgnr_solve_optimal_adjoint(theta, xt, t, y, b_theta, num_iters=20):
    """CGNR: solve a_best = argmin_a || J_C(theta)^T a - b_theta ||^2 by CG
    on the normal equations J_C J_C^T a = J_C b_theta, starting from a=0.

    b_theta: {theta_param_name: FLATTENED (.reshape(-1)) tensor} -- e.g.
      g_cache_direct built per-name from cache_adjoint.compute_g_exact /
      compute_g_semi_and_a_star (both already return flattened per-name
      dicts). MUST be flattened: every intermediate `w`/`z` here comes from
      vjp_cache_to_theta, which always returns .reshape(-1)'d tensors.
    Returns: (a_best: {cache_tensor_name: tensor}, history: list of dicts
      with per-iteration residual_norm, for convergence diagnostics).
    """
    with torch.no_grad():
        z0 = xt.detach().clone()
        _, cache0 = forward_energy_with_cache_grad(theta, z0, t, y)
    cache_shapes = dict(cache0.flatten())

    x = {name: torch.zeros_like(tensor) for name, tensor in cache_shapes.items()}
    r = {k: v.clone() for k, v in b_theta.items()}          # r0 = b - A x0 = b
    z = jvp_theta_to_cache(theta, xt, t, y, r)               # z0 = A^T r0
    p = {k: v.clone() for k, v in z.items()}
    z_dot_z = _dict_dot(z, z)
    _release_autograd_cycles()

    history = []
    for k in range(num_iters):
        w = vjp_cache_to_theta(theta, xt, t, y, p)           # w_k = A p_k
        w_dot_w = _dict_dot(w, w)
        alpha = z_dot_z / (w_dot_w + 1e-30)
        x = {key: x[key] + alpha * p[key] for key in x}
        r = {key: r[key] - alpha * w.get(key, torch.zeros_like(r[key])) for key in r}
        del w
        z_new = jvp_theta_to_cache(theta, xt, t, y, r)       # z_{k+1} = A^T r_{k+1}
        _release_autograd_cycles()
        z_new_dot = _dict_dot(z_new, z_new)
        beta = z_new_dot / (z_dot_z + 1e-30)
        p = {key: z_new[key] + beta * p[key] for key in p}
        z, z_dot_z = z_new, z_new_dot

        history.append({
            "iter": k,
            "residual_norm": _dict_dot(r, r) ** 0.5,
            "alpha": alpha,
            "beta": beta,
        })

    return x, history
