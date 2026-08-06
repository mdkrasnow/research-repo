"""
ForwardBackwardsDirectTrainer: the composite training wrapper that owns
theta (forward scalar EqM), phi (ReverseEqM), the parameter-mapping
registry, and the phi-only optimizer, and implements the reference
algorithm from Section 19 of the spec:

    for each training iteration:
        phi <- Pi(theta)
        with no_grad: E, C = forward_energy_with_cache(theta, z, t, y)
        g_tilde = phi(sg[C])
        L = field_matching_loss(g_tilde, ut)
        L.backward()                      # differentiates ONLY through phi
        phi_before = snapshot(phi)
        optimizer.step()
        delta_phi = phi - phi_before
        theta <- theta + Pi_dagger(delta_phi)
        phi <- Pi(theta)                  # re-sync

theta never appears on the LHS of an autograd graph in this mode -- its
`.grad` stays `None` for every parameter, every step (checked by
tests/test_forward_backwards_direct.py::test_no_double_backward and
enforced by forward_energy_with_cache's `torch.no_grad()`).
"""
import math

import torch

from .cache import TransformerReverseCache
from .forward_cache import forward_energy_with_cache
from .param_mapping import ParameterMappingRegistry
from .reverse_model import ReverseEqM

MAPPING_VERSION = "fb-direct-v1-identity"


def mean_flat(x):
    return x.mean(dim=list(range(1, len(x.shape))))


def _grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().float().norm() ** 2
    return math.sqrt(float(total))


def _param_norm(parameters):
    total = 0.0
    for p in parameters:
        total += p.detach().float().norm() ** 2
    return math.sqrt(float(total))


class ForwardBackwardsDirectTrainer:
    def __init__(self, forward_model, lr=1e-4, weight_decay=0.0, betas=(0.9, 0.999),
                 max_grad_norm=None, device=None):
        if forward_model.ebm != "forward-backwards-direct":
            raise ValueError(
                "ForwardBackwardsDirectTrainer requires a model constructed with "
                f"ebm='forward-backwards-direct', got ebm={forward_model.ebm!r}"
            )
        self.theta = forward_model
        device = device or next(forward_model.parameters()).device
        self.phi = ReverseEqM(forward_model).to(device)
        self.registry = ParameterMappingRegistry(forward_model, self.phi)
        self.frozen_cache_only_names = self.registry.freeze_cache_only_()
        self.registry.tie_from_forward_()
        self.optimizer = torch.optim.AdamW(
            self.phi.parameters(), lr=lr, weight_decay=weight_decay, betas=betas,
        )
        self.max_grad_norm = max_grad_norm
        self.step_count = 0
        self.mapping_version = MAPPING_VERSION

    def coverage_report(self):
        return self.registry.parameter_coverage_report()

    def training_step(self, transport, x1, y, max_grad_norm=None):
        """One full iteration of the reference algorithm. Returns
        (loss: Tensor scalar, diagnostics: dict).
        """
        max_grad_norm = self.max_grad_norm if max_grad_norm is None else max_grad_norm

        t, x0, x1 = transport.sample(x1)
        t = t.to(x1)
        t, xt, ut = transport.path_sampler.plan(t, x0, x1)
        ut = ut * transport.get_ct(t)[:, None, None, None]  # existing repo c(gamma) target

        # 1. hard synchronization phi <- Pi(theta)
        self.registry.tie_from_forward_()
        sync_error_pre = self.registry.compute_sync_error()

        # 2. forward scalar energy + reverse cache, THETA BUILDS NO GRAPH
        E, cache = forward_energy_with_cache(self.theta, xt, t, y)

        # 3. explicit reverse gradient: ordinary forward program in phi.
        # R_phi(cache) computes grad_z E_theta(z) itself (see reverse_model.py:
        # the u=ones seed at the energy-head output IS dE/d(token_energies),
        # propagated with NO sign flip anywhere in reverse_ops.py). The
        # repo's existing convention (models.py:296-300) is that the
        # TRAINING PREDICTION / SAMPLING FIELD is -grad_z E, so negate here.
        grad_tilde = self.phi(cache)
        prediction = -grad_tilde  # matches models.py's `field = -grad E` convention

        per_sample_loss = mean_flat((prediction - ut) ** 2)
        loss = per_sample_loss.mean()

        # 4. backward -- differentiates ONLY through phi (cache tensors are leaves)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        raw_grad_norm = _grad_norm(self.phi.parameters())
        clipped_grad_norm = raw_grad_norm
        if max_grad_norm is not None:
            clipped = torch.nn.utils.clip_grad_norm_(
                self.phi.parameters(), max_grad_norm, error_if_nonfinite=True,
            )
            clipped_grad_norm = float(clipped)

        # 5. snapshot phi, step, compute delta, map into theta, re-sync
        phi_before = self.registry.snapshot_backward_parameters()
        self.optimizer.step()
        theta_update_norms = self.registry.apply_backward_delta_to_forward_(phi_before)
        self.registry.tie_from_forward_()
        sync_error_post = self.registry.compute_sync_error()

        theta_update_norm = math.sqrt(sum(v ** 2 for v in theta_update_norms.values()))
        theta_weight_norm = _param_norm(
            p for name, p in self.theta.named_parameters()
            if name in theta_update_norms
        )

        self.step_count += 1
        diagnostics = {
            "fb/loss": float(loss.detach()),
            "fb/raw_phi_grad_norm": raw_grad_norm,
            "fb/clipped_phi_grad_norm": clipped_grad_norm,
            "fb/theta_update_norm": theta_update_norm,
            "fb/update_to_weight_ratio": (
                theta_update_norm / theta_weight_norm if theta_weight_norm > 0 else 0.0
            ),
            "fb/phi_theta_sync_error_pre": sync_error_pre,
            "fb/phi_theta_sync_error_post": sync_error_post,
            "fb/field_norm": float(prediction.detach().flatten(1).norm(dim=1).mean()),
            "fb/target_norm": float(ut.detach().flatten(1).norm(dim=1).mean()),
            "fb/field_target_cosine": float(
                torch.nn.functional.cosine_similarity(
                    prediction.detach().flatten(1), ut.detach().flatten(1), dim=1,
                ).mean()
            ),
            "fb/energy_mean": float(E.mean()),
        }
        return loss.detach(), diagnostics

    @torch.no_grad()
    def exact_field_audit(self, xt, t, y):
        """Diagnostic-only: compare R_phi(cache) against the TRUE
        autograd.grad(E.sum(), xt) input gradient (both in grad_z E space,
        i.e. NOT the sign-flipped sampling field -- see training_step's
        `grad_tilde`/`prediction` split). Uses
        `torch.autograd.grad(..., create_graph=False)` -- allowed for
        diagnostics per Section 10/13 of the spec -- and must not perturb
        optimizer state (no `.backward()`/`.step()` here).

        Root-cause note (Gate 0 retune #2, 2026-08-06): `forward_energy_with_cache`
        manually recomputes attention as the textbook two-pass
        `softmax(q*scale @ k^T) @ v`, which is bit-for-bit what PyTorch's SDPA
        MATH backend computes (verified to ~1e-16 relative error in FP64), but
        is NOT what timm's `Attention.forward` computes by default on CUDA --
        `use_fused_attn()` is True there, so `self.theta(...)` below would
        otherwise dispatch to whatever fused kernel (flash/mem-efficient)
        PyTorch auto-selects, which is numerically DIFFERENT from two-pass
        softmax at the ~1e-4 relative level per SiT block (same magnitude,
        same depth-compounding profile as the Gate 0 failure), even though
        it's mathematically equivalent. `train.py` already forces the MATH
        backend process-wide whenever `ebm != 'none'` (it has to -- flash
        attention doesn't support `create_graph=True`), but this diagnostic
        module is imported and called independently of `train.py`'s `main()`,
        so that global flag is never guaranteed to be set here. Force it
        locally so the autograd ground truth and the manual cache are
        provably computing through the SAME attention algorithm.
        """
        with torch.enable_grad(), torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            xt_req = xt.detach().clone().requires_grad_(True)
            E_true = self.theta(xt_req, t, y, energy_only=True)
            true_grad = torch.autograd.grad(E_true.sum(), xt_req, create_graph=False)[0]
        true_grad = true_grad.detach()

        _, cache = forward_energy_with_cache(self.theta, xt, t, y)
        approx_grad = self.phi(cache)

        diff = approx_grad - true_grad
        rel_error = diff.flatten(1).norm(dim=1) / true_grad.flatten(1).norm(dim=1).clamp_min(1e-12)
        max_abs_error = diff.abs().max()
        cosine = torch.nn.functional.cosine_similarity(
            approx_grad.flatten(1), true_grad.flatten(1), dim=1
        )
        return {
            "fb/audit_mean_rel_error": float(rel_error.mean()),
            "fb/audit_max_abs_error": float(max_abs_error),
            "fb/audit_mean_cosine": float(cosine.mean()),
        }

    def state_dict(self):
        return {
            "theta": self.theta.state_dict(),
            "phi": self.phi.state_dict(),
            "phi_optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "mapping_version": self.mapping_version,
            "frozen_cache_only_names": self.frozen_cache_only_names,
        }

    def load_state_dict(self, state, strict_theta=True):
        missing, unexpected = self.theta.load_state_dict(state["theta"], strict=strict_theta)
        if missing or unexpected:
            print(f"[forward-backwards-direct] theta load: missing={missing} unexpected={unexpected}")
        if state.get("mapping_version") != self.mapping_version:
            raise RuntimeError(
                f"checkpoint mapping_version={state.get('mapping_version')!r} != "
                f"current {self.mapping_version!r}; refusing to resume with a "
                f"mismatched Pi/Pi-dagger convention."
            )
        if "phi" in state:
            self.phi.load_state_dict(state["phi"])
        else:
            # Loading a bare 'direct' checkpoint into theta only: initialize
            # phi fresh from theta and start a fresh phi optimizer.
            print(
                "[forward-backwards-direct] no phi/optimizer state in checkpoint "
                "(loading from a plain 'direct' checkpoint) -- initializing "
                "phi = Pi(theta) and a fresh phi optimizer."
            )
        self.registry.tie_from_forward_()
        if "phi_optimizer" in state:
            self.optimizer.load_state_dict(state["phi_optimizer"])
        self.step_count = state.get("step_count", 0)
        sync_err = self.registry.compute_sync_error()
        if sync_err > 1e-5:
            raise RuntimeError(
                f"post-load phi/theta sync error {sync_err:.3e} exceeds tolerance"
            )
