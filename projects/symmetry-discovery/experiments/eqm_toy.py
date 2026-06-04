"""
EqM-faithful 2D viability test: can the model fill HELD-OUT manifold regions that a
hidden symmetry implies? Tests whether hard-negative mining (v10 analog) or
symmetry-discovery+constraint is the lever for unknown-rule generalization.

Data   : 8 Gaussian modes equally spaced on a circle. Hidden rule = 45deg rotation.
Split  : train on 6 modes, HOLD OUT 2 (the rule implies they exist).
Model  : EqM skeleton. linear path x_g=(1-g)x + g*eps, predict velocity v=(eps-x).
         (c(g) folded into the linear-path target; faithful to EqM field-matching.)
Sample : integrate dx/dt = -v_pred from noise back to data (Euler).
Metric : recall@heldout = frac of samples landing near a HELD-OUT mode (the gaps),
         and total mode coverage (how many of all 8 modes get >=1% of samples).

Arms:
  BASE       : vanilla EqM on 6 modes.
  HARDNEG    : + v10-style mined hard negatives (PGA to high-loss points), push field.
  EQUIV_KNOWN: + enforce f equivariant under the TRUE 45deg rotation (upper bound).
  EQUIV_DISC : + LEARN the rotation angle from data, then enforce equivariance.
"""
import torch, torch.nn as nn, math

N_MODES   = 8
HELDOUT   = {2, 5}                       # indices of the 2 hidden modes
RADIUS    = 4.0
MODE_STD  = 0.15
STEPS     = 4000
BATCH     = 512
SAMPLE_STEPS = 50
N_SAMPLE  = 2000
SEEDS     = [0, 1, 2]

def mode_centers():
    ang = torch.arange(N_MODES) * (2*math.pi / N_MODES)
    return torch.stack([RADIUS*torch.cos(ang), RADIUS*torch.sin(ang)], 1)   # [8,2]

CENTERS = mode_centers()
TRAIN_MODES = [i for i in range(N_MODES) if i not in HELDOUT]

def sample_data(n, modes):
    idx = torch.tensor(modes)[torch.randint(0, len(modes), (n,))]
    return CENTERS[idx] + MODE_STD * torch.randn(n, 2)

class VNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 2))
    def forward(self, x, g):
        return self.net(torch.cat([x, g], -1))

def eqm_loss(model, x):
    n = x.shape[0]
    g = torch.rand(n, 1)
    eps = torch.randn_like(x)
    xg = (1 - g) * x + g * eps
    target = eps - x                       # EqM linear-path velocity target
    return ((model(xg, g) - target)**2).mean()

def rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)

@torch.no_grad()
def sample(model, n):
    x = torch.randn(n, 2)                   # start at noise (g=1)
    for i in range(SAMPLE_STEPS):
        g = torch.full((n, 1), 1 - i/SAMPLE_STEPS)
        v = model(x, g)
        x = x - v * (1.0/SAMPLE_STEPS)      # integrate toward data
    return x

@torch.no_grad()
def metrics(model):
    s = sample(model, N_SAMPLE)
    d = torch.cdist(s, CENTERS)             # [N,8]
    nearest = d.argmin(1)
    within = (d.min(1).values < 3*MODE_STD)
    counts = torch.zeros(N_MODES)
    for m in range(N_MODES):
        counts[m] = ((nearest == m) & within).float().sum()
    frac = counts / N_SAMPLE
    recall_heldout = frac[list(HELDOUT)].sum().item()
    modes_covered = int((frac > 0.01).sum().item())
    return recall_heldout, modes_covered, frac

def train(arm, seed):
    torch.manual_seed(seed)
    model = VNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    theta = None
    if arm == "EQUIV_DISC":
        theta = torch.tensor([0.5], requires_grad=True)  # init off-identity (~28deg)
        opt = torch.optim.Adam(list(model.parameters()) + [theta], lr=2e-3)
    true_R = rot(2*math.pi / N_MODES)                   # 45 deg

    for step in range(STEPS):
        x = sample_data(BATCH, TRAIN_MODES)
        opt.zero_grad()
        loss = eqm_loss(model, x)

        if arm == "HARDNEG":
            # v10 analog: mine hard points by PGA on field-matching residual, push field there
            g = torch.rand(BATCH, 1)
            eps = torch.randn_like(x)
            xg = ((1-g)*x + g*eps).detach().requires_grad_(True)
            tgt = (eps - x).detach()
            r = ((model(xg, g) - tgt)**2).sum(1).mean()
            grad = torch.autograd.grad(r, xg, create_graph=False)[0]
            xg_hard = (xg + 0.5*grad.sign()).detach()    # PGA step to harder input
            loss = loss + ((model(xg_hard, g) - tgt)**2).mean()

        if arm == "EQUIV_KNOWN":
            g = torch.rand(BATCH, 1)
            eps = torch.randn_like(x); xg = (1-g)*x + g*eps
            # f equivariant: f(R x) = R f(x)
            fx = model(xg, g)
            fRx = model(xg @ true_R.T, g)
            loss = loss + ((fRx - fx @ true_R.T)**2).mean()

        if arm == "EQUIV_DISC":
            R = rot_param(theta)
            x2 = sample_data(BATCH, TRAIN_MODES)
            # (a) closure: rotating real data stays real -> low eqm loss under R
            loss = loss + 3.0 * eqm_loss_rotated(model, x2, R)
            # (b) non-degeneracy BARRIER: forbid identity (theta near 0/2pi).
            #     sin^2(theta/2) = (1-cos)/2 -> 0 at identity; 1/(.) blows up there.
            half = (1 - torch.cos(theta[0])) / 2
            loss = loss + 0.05 / (half + 0.05)
            # (c) equivariance under learned R
            g = torch.rand(BATCH, 1)
            eps = torch.randn_like(x); xg = (1-g)*x + g*eps
            fx = model(xg, g); fRx = model(xg @ R.T, g)
            loss = loss + ((fRx - fx @ R.T)**2).mean()

        loss.backward(); opt.step()

    rh, mc, frac = metrics(model)
    extra = ""
    if arm == "EQUIV_DISC":
        deg = math.degrees(theta.item()) % 360
        extra = f"  learned_angle={deg:.1f}deg (true 45)"
    return rh, mc, frac, extra

def rot_param(theta):
    c, s = torch.cos(theta[0]), torch.sin(theta[0])
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

def eqm_loss_rotated(model, x, R):
    xr = x @ R.T
    n = xr.shape[0]; g = torch.rand(n, 1); eps = torch.randn_like(xr)
    xg = (1-g)*xr + g*eps; target = eps - xr
    return ((model(xg, g) - target)**2).mean()

print(f"held-out modes = {sorted(HELDOUT)} of {N_MODES}")
print(f"{'arm':12s} {'recall@heldout':>14s} {'modes_cov/8':>12s}")
print("-" * 42)
for arm in ("BASE", "HARDNEG", "EQUIV_KNOWN", "EQUIV_DISC"):
    rhs, mcs, ex = [], [], ""
    for s in SEEDS:
        rh, mc, frac, extra = train(arm, s); rhs.append(rh); mcs.append(mc); ex = extra
    mrh = sum(rhs)/len(rhs); mmc = sum(mcs)/len(mcs)
    sd = (sum((x-mrh)**2 for x in rhs)/len(rhs))**0.5
    print(f"{arm:12s} {mrh:8.3f}±{sd:.3f} {mmc:10.1f}{ex}")
print("\nideal recall@heldout = 0.25 (2/8 modes).  BASE expected ~0.0")
