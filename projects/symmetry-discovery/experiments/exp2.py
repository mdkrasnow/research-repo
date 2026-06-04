"""
Corrected viability test. Fix v1 bugs:
  - TRUE overfit: full seen-region negatives (all c != a+b) so train_acc -> ~1.
  - Add the lever I called "overengineering": EQUIV (tie E(a,b,c)=E(b,a,c)).
  - Add FULLNEG upper bound (full swapped complement push-up = elimination = ~leak).

Find which lever actually carries generalization to unseen orderings (a>b).

Arms (all share BASE = seen positives + full seen negatives -> overfit seen region):
  BASE     : nothing extra.                                  expect test ~ chance
  STRUCT   : + sparse plausible swapped negs (b,a,c-) K=4.   the minimal thesis
  FULLNEG  : + full swapped complement (b,a, all c!=a+b).    upper bound (~leak by elim)
  EQUIV    : + lambda*||E(a,b,c)-E(b,a,c)||^2.               the equivariance lever
"""
import torch, torch.nn as nn, random

NA = NB = 10
NC = 19
K  = 4
STEPS = 3000
SEEDS = [0, 1, 2]
EQ_LAMBDA = 1.0

def onehot(i, n):
    v = torch.zeros(n); v[i] = 1.0; return v
def feat(a, b, c):
    return torch.cat([onehot(a, NA), onehot(b, NB), onehot(c, NC)])
ALL_C = list(range(NC))

class EnergyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NA+NB+NC, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

train_pairs = [(a, b) for a in range(NA) for b in range(NB) if a <= b]
test_pairs  = [(a, b) for a in range(NA) for b in range(NB) if a >  b]

def build(arm, rng):
    pos, neg, eqA, eqB = [], [], [], []
    for (a, b) in train_pairs:
        c = a + b
        pos.append(feat(a, b, c))
        for cc in ALL_C:                       # full seen negatives -> real overfit
            if cc != c: neg.append(feat(a, b, cc))
        if arm == "STRUCT":
            for _ in range(K):
                cc = rng.randrange(NC)
                while cc == c: cc = rng.randrange(NC)
                neg.append(feat(b, a, cc))
        if arm == "FULLNEG":
            for cc in ALL_C:
                if cc != c: neg.append(feat(b, a, cc))
        if arm == "EQUIV":
            for cc in ALL_C:                   # tie energies across the swap, all c
                eqA.append(feat(a, b, cc)); eqB.append(feat(b, a, cc))
    eq = (torch.stack(eqA), torch.stack(eqB)) if eqA else None
    return torch.stack(pos), torch.stack(neg), eq

@torch.no_grad()
def accuracy(model, pairs):
    ok = 0
    for (a, b) in pairs:
        es = model(torch.stack([feat(a, b, c) for c in ALL_C]))
        if ALL_C[int(es.argmin())] == a + b: ok += 1
    return ok / len(pairs)

def run(arm, seed):
    torch.manual_seed(seed); rng = random.Random(seed)
    Xp, Xn, eq = build(arm, rng)
    model = EnergyMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(STEPS):
        opt.zero_grad()
        lp = -model(Xp); ln = -model(Xn)
        loss = bce(lp, torch.ones_like(lp)) + bce(ln, torch.zeros_like(ln))
        if eq is not None:
            loss = loss + EQ_LAMBDA * ((model(eq[0]) - model(eq[1]))**2).mean()
        loss.backward(); opt.step()
    return accuracy(model, train_pairs), accuracy(model, test_pairs)

print(f"{'arm':9s} {'train(a<=b)':>12s} {'TEST(a>b)':>14s}")
print("-" * 38)
for arm in ("BASE", "STRUCT", "FULLNEG", "EQUIV"):
    trs, tes = [], []
    for s in SEEDS:
        tr, te = run(arm, s); trs.append(tr); tes.append(te)
    mt = sum(trs)/len(trs); me = sum(tes)/len(tes)
    sd = (sum((x-me)**2 for x in tes)/len(tes))**0.5
    print(f"{arm:9s} {mt:12.3f} {me:8.3f}±{sd:.3f}")
print("\nchance = %.3f" % (1/NC))
