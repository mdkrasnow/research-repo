"""
Ladder rung 3: latent symmetry hidden by a nonlinear decoder.

Latent ring (8 modes, 45deg rotation symmetry) -> FROZEN random nonlinear decoder g
-> observed data in R^D where the symmetry is a nonlinear warp (g o R o g^-1), NOT a
named linear op. Train EqM in observed space. Can a learned operator T_psi discover and
enforce the hidden manifold-preserving transformation, with no access to g, z, or R?

Trimmed design (4 arms, 5 core metrics, 2-term discovery recipe):
  BASE         vanilla EqM on 6 modes                         (floor)
  ORACLE       + aug on decode(R_true . z)  (we have z)       (upper bound)
  DISC_LINEAR  learn T(x)=Ax+b, discovery recipe              (control: linear must FAIL)
  DISC_NONLIN  learn residual-MLP T, discovery recipe         (the test)

Discovery recipe = base EqM + W_AUG*eqm(f,T(x)) + W_ID/||T(x)-x||^2 ; weight-decay = smoothness.

Decoder + data are FIXED across all arms/seeds (seed 1234). Only field/T init varies by seed.
Device: CUDA if usable, else CPU (auto-fallback; tiny either way).
"""
import argparse, json, math, os
import torch, torch.nn as nn

# ---------------- config ----------------
D            = 16
N_MODES      = 8
HELDOUT      = [2, 5]
RADIUS       = 4.0
SIG_LAT      = 0.08
STEPS        = 4000
BATCH        = 512
SAMPLE_STEPS = 50
N_SAMPLE     = 2000
SEEDS        = [0, 1, 2]
W_AUG        = 1.0
W_MOVE       = 0.5     # pull ||T(x)-x|| toward ~one-mode spacing (not identity, not infinity)
W_CYC        = 1.0     # finite-order constraint T^P ~ id -> orbit generator, anti-drift
P_CYCLE      = 8       # coarse structural prior: hidden symmetry has order ~8 (NOT the symmetry itself)
W_WD         = 1e-4
DEC_SEED     = 1234
ARMS         = ["BASE", "ORACLE", "DISC_LINEAR", "DISC_NONLIN"]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_MODES = [i for i in range(N_MODES) if i not in HELDOUT]


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    layers += [nn.Linear(sizes[-2], sizes[-1])]
    return nn.Sequential(*layers)


# ---------------- frozen nonlinear decoder + fixed data manifold ----------------
def build_world():
    g = torch.Generator(device="cpu").manual_seed(DEC_SEED)
    dec = nn.Sequential(nn.Linear(2, 64), nn.Tanh(),
                        nn.Linear(64, 64), nn.Tanh(),
                        nn.Linear(64, D))
    with torch.no_grad():
        for p in dec.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (1.0 if p.dim() == 1 else 0.9))
    dec = dec.to(DEV)
    for p in dec.parameters():
        p.requires_grad_(False)

    ang = torch.arange(N_MODES, device=DEV) * (2 * math.pi / N_MODES)
    centers = torch.stack([RADIUS * torch.cos(ang), RADIUS * torch.sin(ang)], 1)  # [8,2]

    # standardization stats over the FULL 8-mode distribution (fixed affine)
    with torch.no_grad():
        zc = centers[torch.randint(0, N_MODES, (20000,), device=DEV)]
        zc = zc + SIG_LAT * torch.randn(20000, 2, device=DEV)
        raw = dec(zc)
        mu, sd = raw.mean(0), raw.std(0) + 1e-6
    decode = lambda z: (dec(z) - mu) / sd

    with torch.no_grad():
        M_obs = decode(centers)                                   # [8,D] anchors (eval only)
        # observed noise scale around anchors
        zc = centers[2].repeat(4000, 1) + SIG_LAT * torch.randn(4000, 2, device=DEV)
        sig_obs = (decode(zc) - M_obs[2]).norm(dim=1).mean().item()
        pair = torch.cdist(M_obs, M_obs)
        pair = pair + torch.eye(N_MODES, device=DEV) * 1e9
        min_pair = pair.min().item()
    return decode, centers, M_obs, sig_obs, min_pair


def rot_true():
    t = 2 * math.pi / N_MODES
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=DEV)


# ---------------- EqM field ----------------
def eqm_loss(f, x):
    n = x.shape[0]
    g = torch.rand(n, 1, device=DEV)
    eps = torch.randn(n, D, device=DEV)
    xg = (1 - g) * x + g * eps
    return ((f(torch.cat([xg, g], 1)) - (eps - x)) ** 2).mean()


@torch.no_grad()
def sample(f, n):
    x = torch.randn(n, D, device=DEV)
    for i in range(SAMPLE_STEPS):
        g = torch.full((n, 1), 1 - i / SAMPLE_STEPS, device=DEV)
        x = x - f(torch.cat([x, g], 1)) * (1.0 / SAMPLE_STEPS)
    return x


# ---------------- operators ----------------
class LinearT(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.eye(D) + 0.05 * torch.randn(D, D))
        self.b = nn.Parameter(0.05 * torch.randn(D))
    def forward(self, x): return x @ self.A.T + self.b

class ResidualT(nn.Module):
    def __init__(self, s=0.3):
        super().__init__()
        self.h = mlp([D, 128, 128, D])
        self.s = s
    def forward(self, x): return x + self.s * self.h(x)


# ---------------- world (built once) ----------------
DECODE, CENTERS, M_OBS, SIG_OBS, MIN_PAIR = build_world()
R_TRUE = rot_true()


def sample_latent(n):
    idx = torch.tensor(TRAIN_MODES, device=DEV)[torch.randint(0, len(TRAIN_MODES), (n,), device=DEV)]
    return CENTERS[idx] + SIG_LAT * torch.randn(n, 2, device=DEV)


def mode_of(x):                       # nearest observed-mode index per row
    return torch.cdist(x, M_OBS).argmin(1)


# ---------------- train one arm/seed ----------------
def train(arm, seed):
    torch.manual_seed(seed)
    if DEV.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    f = mlp([D + 1, 256, 256, D]).to(DEV)
    T = None
    params = list(f.parameters())
    wd = 0.0
    if arm == "DISC_LINEAR":
        T = LinearT().to(DEV); params += list(T.parameters()); wd = W_WD
    elif arm == "DISC_NONLIN":
        T = ResidualT().to(DEV); params += list(T.parameters()); wd = W_WD
    opt = torch.optim.Adam(params, lr=2e-3, weight_decay=wd)

    for _ in range(STEPS):
        z = sample_latent(BATCH)
        x = DECODE(z)
        opt.zero_grad()
        loss = eqm_loss(f, x)
        if arm == "ORACLE":
            loss = loss + W_AUG * eqm_loss(f, DECODE(z @ R_TRUE.T))
        elif arm in ("DISC_LINEAR", "DISC_NONLIN"):
            Tx = T(x)
            loss = loss + W_AUG * eqm_loss(f, Tx)                       # closure/aug
            move = (Tx - x).pow(2).sum(1).clamp_min(1e-8).sqrt()
            loss = loss + W_MOVE * (move - MIN_PAIR).pow(2).mean()      # ~one-mode move
            xk = x                                                       # finite-order: T^P ~ id
            for _ in range(P_CYCLE):
                xk = T(xk)
            loss = loss + W_CYC * (xk - x).pow(2).sum(1).mean()
        loss.backward(); opt.step()
    return f, T


# ---------------- metrics ----------------
def energy_distance(a, b):
    daa = torch.cdist(a, a).mean(); dbb = torch.cdist(b, b).mean()
    dab = torch.cdist(a, b).mean()
    return (2 * dab - daa - dbb).item()


@torch.no_grad()
def evaluate(f, T):
    s = sample(f, N_SAMPLE)
    d = torch.cdist(s, M_OBS)
    near = d.argmin(1); within = d.min(1).values < 3 * SIG_OBS
    frac = torch.zeros(N_MODES, device=DEV)
    for m in range(N_MODES):
        frac[m] = ((near == m) & within).float().sum()
    frac = frac / N_SAMPLE
    ref = DECODE(CENTERS[torch.randint(0, N_MODES, (N_SAMPLE,), device=DEV)]
                 + SIG_LAT * torch.randn(N_SAMPLE, 2, device=DEV))
    out = {
        "recall_heldout": float(frac[HELDOUT].sum()),
        "modes_covered": int((frac > 0.01).sum()),
        "mmd_energy": energy_distance(s, ref),
    }
    if T is not None:
        xb = DECODE(sample_latent(512))
        Tx = T(xb)
        out["T_move_ratio"] = float((Tx - xb).norm(dim=1).mean() / MIN_PAIR)
        out["T_on_manifold"] = float((torch.cdist(Tx, M_OBS).min(1).values < 3 * SIG_OBS).float().mean())
        out["T_mode_change"] = float((mode_of(Tx) != mode_of(xb)).float().mean())
        sigma = {}
        for i in TRAIN_MODES:
            xi = DECODE(CENTERS[i].repeat(128, 1) + SIG_LAT * torch.randn(128, 2, device=DEV))
            j = torch.bincount(mode_of(T(xi)), minlength=N_MODES).argmax().item()
            sigma[i] = int(j)
        out["sigma"] = sigma
        out["hits_heldout"] = bool(any(v in HELDOUT for v in sigma.values()))
        shifts = [(sigma[i] - i) % N_MODES for i in TRAIN_MODES]
        out["consistent_shift"] = (len(set(shifts)) == 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="results.json")
    args = ap.parse_args()

    print(f"device={DEV}  D={D}  modes={N_MODES}  heldout={HELDOUT}")
    print(f"decoder injectivity: min_pair={MIN_PAIR:.3f}  sig_obs={SIG_OBS:.3f}  "
          f"ratio={MIN_PAIR/SIG_OBS:.1f} (want >>3)")
    print(f"ideal recall_heldout = {len(HELDOUT)/N_MODES:.3f}\n")

    results = {"config": {"D": D, "heldout": HELDOUT, "steps": STEPS, "seeds": SEEDS,
                          "min_pair": MIN_PAIR, "sig_obs": SIG_OBS, "device": str(DEV)},
               "arms": {}}
    hdr = f"{'arm':12s} {'recall@HO':>10s} {'modes/8':>8s} {'mmd':>8s} {'T_move':>7s} {'T_onman':>8s} {'hits_HO':>8s} {'cyclic':>7s}"
    print(hdr); print("-" * len(hdr))
    for arm in ARMS:
        runs = [evaluate(*train(arm, s)) for s in SEEDS]
        agg = {}
        for k in ("recall_heldout", "modes_covered", "mmd_energy",
                  "T_move_ratio", "T_on_manifold", "T_mode_change"):
            vals = [r[k] for r in runs if k in r]
            if vals: agg[k] = sum(vals) / len(vals)
        agg["recall_sd"] = (sum((r["recall_heldout"] - agg["recall_heldout"]) ** 2
                                for r in runs) / len(runs)) ** 0.5
        agg["hits_heldout_any"] = any(r.get("hits_heldout") for r in runs)
        agg["consistent_shift_any"] = any(r.get("consistent_shift") for r in runs)
        agg["sigma_seed0"] = runs[0].get("sigma")
        results["arms"][arm] = agg
        rh = agg["recall_heldout"]; sd = agg["recall_sd"]
        tm = agg.get("T_move_ratio"); to = agg.get("T_on_manifold")
        print(f"{arm:12s} {rh:6.3f}±{sd:.3f} {agg['modes_covered']:8.1f} "
              f"{agg['mmd_energy']:8.3f} "
              f"{('%.3f'%tm) if tm is not None else '   -  ':>7s} "
              f"{('%.3f'%to) if to is not None else '   -  ':>8s} "
              f"{str(agg['hits_heldout_any']):>8s} {str(agg['consistent_shift_any']):>7s}")
    print(f"\nideal recall@HO = {len(HELDOUT)/N_MODES:.3f}")
    print("sigma (seed0) DISC_NONLIN:", results["arms"]["DISC_NONLIN"].get("sigma_seed0"))

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
