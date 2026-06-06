"""v14 (beat-crop) Rung A — policy safety: autograd to POLICY params, encoder frozen, no move leakage.

The policy is a distribution qθ over a 2-generator SE(2) Lie basis:
    M = exp(z1*A1 + z2*A2),  z ~ N(0, diag(exp(logsig)^2))     (params: A1,A2 [2x3], logsig [2])
Gate (before any utility work): anchor energy-dist loss must back-prop to ALL policy params while the
frozen encoder stays frozen; the @no_grad path must give zero grad; a BROAD move hinge must learn from the
anchor without a leaked target magnitude.
"""
import sys, os
import torch
from torchvision import datasets, transforms
sys.path.insert(0, os.path.dirname(__file__))
from _se2_discovery import FrozenConv, warp, trans_M, _ed_t  # noqa

DEV = torch.device("cpu")
Z = torch.zeros(1, 3, device=DEV)


def load_cifar(n=200):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    root = next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data"))
                 if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))), "data")
    ds = datasets.CIFAR10(root, train=True, download=False, transform=tf)
    return torch.stack([ds[i][0] for i in torch.randperm(len(ds))[:n].tolist()])


def sample_M(A1, A2, logsig):
    z = torch.randn(2, device=DEV) * torch.exp(logsig)
    return torch.matrix_exp(z[0] * torch.cat([A1, Z]) + z[1] * torch.cat([A2, Z]))


def main():
    torch.manual_seed(0)
    tf_imgs = load_cifar(200).to(DEV)   # real images: translation magnitude grows with shift (move band valid)
    enc = FrozenConv().to(DEV)
    n_frozen = sum(1 for p in enc.parameters() if not p.requires_grad)
    n_params = sum(1 for _ in enc.parameters())

    def policy_anchor_grad(use_grad):
        A1 = torch.nn.Parameter(0.05*torch.randn(2,3)); A2 = torch.nn.Parameter(0.05*torch.randn(2,3))
        logsig = torch.nn.Parameter(torch.tensor([-0.5,-0.5]))
        M = sample_M(A1, A2, logsig); Tx = warp(tf_imgs, M)
        fT = enc.fg(Tx) if use_grad else enc(Tx)
        loss = _ed_t(fT, enc.fg(tf_imgs).detach())
        try:
            gs = torch.autograd.grad(loss, [A1, A2, logsig], allow_unused=True)
        except RuntimeError:
            gs = [None, None, None]
        gn = sum(float(g.abs().sum()) for g in gs if g is not None)
        return gn, all(g is not None for g in gs[:2])

    grad_norm, grad_ok = policy_anchor_grad(True)
    nog_norm, nog_ok = policy_anchor_grad(False)

    # move regimes: broad hinge (anchor-driven) vs leaked target (handed magnitude)
    with torch.no_grad():
        m2 = (warp(tf_imgs, trans_M(2,0,DEV))-tf_imgs).flatten(1).norm(dim=1).mean()
        m4 = (warp(tf_imgs, trans_M(4,0,DEV))-tf_imgs).flatten(1).norm(dim=1).mean()
        m10 = (warp(tf_imgs, trans_M(10,0,DEV))-tf_imgs).flatten(1).norm(dim=1).mean()
    def broad(mv): return torch.relu((m2-mv)/m2)**2 + torch.relu((mv-m10)/m2)**2
    def leaked(mv): return ((mv-m4)/m4)**2

    def learn(move_mode, steps=200):
        torch.manual_seed(1)
        A1=torch.nn.Parameter(0.02*torch.randn(2,3)); A2=torch.nn.Parameter(0.02*torch.randn(2,3))
        logsig=torch.nn.Parameter(torch.tensor([-1.0,-1.0]))
        opt=torch.optim.Adam([A1,A2,logsig],lr=5e-3)
        anchor = enc(tf_imgs)
        for s in range(steps):
            M=sample_M(A1,A2,logsig); Tx=warp(tf_imgs[:64],M); L=M[:2,:2]
            mv=(Tx-tf_imgs[:64]).flatten(1).norm(dim=1).mean(); sv=torch.linalg.svdvals(L)
            f=torch.cat([enc.fg(tf_imgs[:64]).detach(), enc.fg(Tx)],0)
            pen = broad(mv) if move_mode=="broad" else leaked(mv)
            loss=_ed_t(f, anchor[torch.randperm(anchor.size(0))[:96]]) + 0.5*pen + (sv.max()/sv.min().clamp_min(1e-6)-1)**2
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            mvf=float((warp(tf_imgs, sample_M(A1.detach(),A2.detach(),logsig.detach()))-tf_imgs).flatten(1).norm(dim=1).mean())
        return mvf
    broad_move = learn("broad"); leaked_move = learn("leaked")

    print("=== v14 (beat-crop) Rung A — policy safety ===")
    print(f"encoder frozen: {n_frozen}/{n_params}")
    print(f"GRAD_ANCHOR -> policy grad nonzero: {grad_ok} | grad_norm={grad_norm:.4e}")
    print(f"NO_GRAD_ANCHOR -> grad exists: {nog_ok} | grad_norm={nog_norm:.4e} (must be 0)")
    print(f"move band: 2px={float(m2):.1f} 4px(LEAK)={float(m4):.1f} 10px={float(m10):.1f}")
    print(f"BROAD learned move={broad_move:.1f} (in band {float(m2):.1f}-{float(m10):.1f}) | LEAKED move={leaked_move:.1f} (~4px handed)")
    grad_pass = grad_ok and grad_norm>1e-8 and (not nog_ok) and nog_norm<1e-12
    broad_pass = float(m2)*0.6 <= broad_move <= float(m10)*1.4
    leaked_invalid = abs(leaked_move-float(m4))/float(m4) < 0.4
    ok = grad_pass and broad_pass and leaked_invalid and n_frozen==n_params
    print(f"\ngrad gate:{grad_pass} broad-learns:{broad_pass} leaked-invalid:{leaked_invalid}")
    print("RUNG A:", "PASS — anchor grad reaches full policy (A1,A2,logsig); encoder frozen; broad hinge no-leak"
          if ok else "FAIL — STOP (policy grad zero / encoder unfrozen / hinge mis-behaves)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
