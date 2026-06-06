"""v14 Rung A — anchor-gradient gate.

The v12 bug was a @torch.no_grad encoder that silently zeroed the anchor gradient to the operator (so
"discovery" was driven only by move+stability). Before any v14 distribution work, GATE that:
  - the frozen encoder's PARAMS stay frozen (requires_grad False, unchanged after a step), AND
  - the anchor (energy-distance) loss has NONZERO gradient to the operator generator A via the grad-flowing
    feature path.
Negative control NO_GRAD_ANCHOR uses the @no_grad path and MUST give ~zero / no grad (reproduce the bug).

PASS: GRAD_ANCHOR grad-norm > 0 (well above 0); NO_GRAD_ANCHOR grad is None/zero; params unchanged.
Stop the ladder if GRAD_ANCHOR grad is zero.
"""
import sys, os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from feature_gap_proxy_cifar_se2 import FrozenConv, affine_warp3, trans_mat3, ed_t  # noqa

DEV = torch.device("cpu")


def build_M(A2):
    A = torch.cat([A2, torch.zeros(1, 3, device=DEV)], 0)
    return torch.matrix_exp(A)


def main():
    torch.manual_seed(0)
    enc = FrozenConv().to(DEV)
    x = torch.randn(64, 3, 32, 32)
    real = torch.randn(64, 3, 32, 32)

    # params frozen?
    n_params = sum(1 for _ in enc.parameters())
    n_frozen = sum(1 for p in enc.parameters() if not p.requires_grad)
    p0 = [p.detach().clone() for p in enc.parameters()]

    def anchor_loss(use_grad):
        A2 = torch.nn.Parameter(0.05 * torch.randn(2, 3, device=DEV))
        M = build_M(A2)
        Tx = affine_warp3(x, M)
        fT = enc.feat_grad(Tx) if use_grad else enc(Tx)   # enc() is @no_grad
        fR = enc.feat_grad(real) if use_grad else enc(real)
        # detach anchor side either way (frozen reference); only T-side carries operator grad
        loss = ed_t(fT, fR.detach())
        try:
            g = torch.autograd.grad(loss, A2, allow_unused=True)[0]
        except RuntimeError:
            g = None
        gn = float(g.abs().sum()) if g is not None else 0.0
        return gn, (g is not None)

    grad_norm, grad_exists = anchor_loss(True)
    nog_norm, nog_exists = anchor_loss(False)

    # take an optimizer step on A with the grad path; confirm encoder params unchanged
    A2 = torch.nn.Parameter(0.05 * torch.randn(2, 3, device=DEV))
    opt = torch.optim.Adam([A2], lr=1e-2)
    for _ in range(5):
        M = build_M(A2); Tx = affine_warp3(x, M)
        loss = ed_t(enc.feat_grad(Tx), enc.feat_grad(real).detach())
        opt.zero_grad(); loss.backward(); opt.step()
    params_unchanged = all(torch.equal(a, b) for a, b in zip(p0, enc.parameters()))

    print("=== v14 Rung A — anchor-gradient gate ===")
    print(f"encoder params: {n_frozen}/{n_params} frozen (requires_grad=False)")
    print(f"GRAD_ANCHOR     grad_to_A nonzero? {grad_exists} | grad_norm = {grad_norm:.6e}")
    print(f"NO_GRAD_ANCHOR  grad_to_A exists?  {nog_exists} | grad_norm = {nog_norm:.6e}   (neg ctrl: must be ~0/None)")
    print(f"encoder params unchanged after 5 operator steps? {params_unchanged}")
    ok = (grad_exists and grad_norm > 1e-8) and (not nog_exists or nog_norm < 1e-12) \
         and (n_frozen == n_params) and params_unchanged
    print("\nRUNG A:", "PASS — anchor signal trains the operator; encoder stays frozen" if ok
          else "FAIL — STOP ladder (anchor grad zero or encoder not frozen)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
