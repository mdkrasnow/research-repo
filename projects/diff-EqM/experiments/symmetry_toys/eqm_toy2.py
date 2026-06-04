"""
Controlled version. Separate the two ways a symmetry can be used, equal weight (W),
to find which mechanism fills held-out modes, and to compare discovery vs known FAIRLY.

  CONSTRAINT : ||f(Rx) - R f(x)||^2          (tie field across the symmetry)
  AUGMENT    : EqM loss on rotated data        (= data-aug into the gaps)

Arms (W=1.0 for every extra term):
  BASE             : vanilla EqM on 6 modes
  HARDNEG          : + v10-style mined hard negatives
  KNOWN_CONSTRAINT : + constraint only            (true 45deg R)
  KNOWN_AUG        : + augment only               (true 45deg R)
  KNOWN_BOTH       : + constraint + augment        (true 45deg R)
  DISC_BOTH        : LEARN R, + constraint + augment   (fair vs KNOWN_BOTH)
"""
import torch, torch.nn as nn, math

N_MODES   = 8
HELDOUT   = {2, 5}
RADIUS    = 4.0
MODE_STD  = 0.15
STEPS     = 4000
BATCH     = 512
SAMPLE_STEPS = 50
N_SAMPLE  = 2000
SEEDS     = [0, 1, 2]
W         = 1.0                              # equal weight on every extra term

ang = torch.arange(N_MODES) * (2*math.pi / N_MODES)
CENTERS = torch.stack([RADIUS*torch.cos(ang), RADIUS*torch.sin(ang)], 1)
TRAIN_MODES = [i for i in range(N_MODES) if i not in HELDOUT]

def sample_data(n):
    idx = torch.tensor(TRAIN_MODES)[torch.randint(0, len(TRAIN_MODES), (n,))]
    return CENTERS[idx] + MODE_STD * torch.randn(n, 2)

class VNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 2))
    def forward(self, x, g): return self.net(torch.cat([x, g], -1))

def eqm_loss_on(model, x):
    n = x.shape[0]; g = torch.rand(n, 1); eps = torch.randn_like(x)
    xg = (1-g)*x + g*eps
    return ((model(xg, g) - (eps - x))**2).mean()

def true_R():
    t = 2*math.pi / N_MODES; c, s = math.cos(t), math.sin(t)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)

def rot_param(theta):
    c, s = torch.cos(theta[0]), torch.sin(theta[0])
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

def constraint_term(model, x, R):
    g = torch.rand(x.shape[0], 1); eps = torch.randn_like(x); xg = (1-g)*x + g*eps
    fx = model(xg, g); fRx = model(xg @ R.T, g)
    return ((fRx - fx @ R.T)**2).mean()

@torch.no_grad()
def sample(model, n):
    x = torch.randn(n, 2)
    for i in range(SAMPLE_STEPS):
        g = torch.full((n, 1), 1 - i/SAMPLE_STEPS)
        x = x - model(x, g) * (1.0/SAMPLE_STEPS)
    return x

@torch.no_grad()
def metrics(model):
    s = sample(model, N_SAMPLE); d = torch.cdist(s, CENTERS)
    nearest = d.argmin(1); within = d.min(1).values < 3*MODE_STD
    frac = torch.tensor([((nearest == m) & within).float().sum() for m in range(N_MODES)]) / N_SAMPLE
    return frac[list(HELDOUT)].sum().item(), int((frac > 0.01).sum().item())

def train(arm, seed):
    torch.manual_seed(seed)
    model = VNet()
    discover = (arm == "DISC_BOTH")
    use_con  = arm in ("KNOWN_CONSTRAINT", "KNOWN_BOTH", "DISC_BOTH")
    use_aug  = arm in ("KNOWN_AUG", "KNOWN_BOTH", "DISC_BOTH")
    params = list(model.parameters())
    theta = None
    if discover:
        theta = torch.tensor([0.5], requires_grad=True); params = params + [theta]
    opt = torch.optim.Adam(params, lr=2e-3)
    R_fixed = true_R()

    for _ in range(STEPS):
        x = sample_data(BATCH)
        opt.zero_grad()
        loss = eqm_loss_on(model, x)
        R = rot_param(theta) if discover else R_fixed

        if arm == "HARDNEG":
            g = torch.rand(BATCH, 1); eps = torch.randn_like(x)
            xg = ((1-g)*x + g*eps).detach().requires_grad_(True); tgt = (eps-x).detach()
            r = ((model(xg, g) - tgt)**2).sum(1).mean()
            grad = torch.autograd.grad(r, xg)[0]
            xg_hard = (xg + 0.5*grad.sign()).detach()
            loss = loss + W * ((model(xg_hard, g) - tgt)**2).mean()
        if use_aug:
            loss = loss + W * eqm_loss_on(model, sample_data(BATCH) @ R.T)
        if use_con:
            loss = loss + W * constraint_term(model, x, R)
        if discover:                                   # identity barrier (only to learn R)
            half = (1 - torch.cos(theta[0])) / 2
            loss = loss + 0.05 / (half + 0.05)

        loss.backward(); opt.step()

    rh, mc = metrics(model)
    extra = ""
    if discover:
        extra = f"  learned={math.degrees(theta[0].item())%360:.1f}deg"
    return rh, mc, extra

ARMS = ("BASE", "HARDNEG", "KNOWN_CONSTRAINT", "KNOWN_AUG", "KNOWN_BOTH", "DISC_BOTH")
print(f"held-out = {sorted(HELDOUT)} of {N_MODES}.  W={W}.  ideal recall=0.25")
print(f"{'arm':18s} {'recall@heldout':>14s} {'modes/8':>8s}")
print("-" * 46)
for arm in ARMS:
    rhs, mcs, ex = [], [], ""
    for s in SEEDS:
        rh, mc, e = train(arm, s); rhs.append(rh); mcs.append(mc); ex = e
    m = sum(rhs)/len(rhs); sd = (sum((x-m)**2 for x in rhs)/len(rhs))**0.5
    print(f"{arm:18s} {m:8.3f}±{sd:.3f} {sum(mcs)/len(mcs):7.1f}{ex}")
