"""
Ladder rung 4: is the hidden symmetry discoverable as a SIMPLE operator in LEARNED
LATENT coordinates, even though it is not discoverable as a free observed-space operator
(rung 3 result: DISC_NONLIN failed, recall 0)?

Identical world/data/split/metrics/seeds as rung 3. ONLY new thing: a 5th arm
  DISC_LATENT : T_psi(x) = dec(M . enc(x)),  M a small learned 2x2 matrix in a learned latent.
The discovery LOSS RECIPE is held FIXED vs rung-3 DISC_NONLIN (closure-aug + bounded-move +
cyclic T^P~=I). So any difference is due to the PARAMETRIZATION, not added optimization pressure.

Controlled-experiment discipline:
  - BASE        negative/floor      (must fail)
  - ORACLE      positive control    (must fill gap -> harness valid)
  - DISC_LINEAR negative control    (linear obs-space op must fail on nonlinear manifold)
  - DISC_NONLIN rung-3 treatment    (free obs-space op; expected fail)
  - DISC_LATENT THE TREATMENT       (enc-linear-dec)

AE protocol: pretrain enc/dec on reconstruction of the 6 TRAIN modes; REPORT recon error
BEFORE judging symmetry. Then semi-freeze (keep a recon anchor) while learning M + field.
Anti-memorization: M is ONE global 2x2 matrix (cannot per-sample look up seen->heldout);
recon anchor keeps enc/dec faithful; held-out modes NEVER enter training data.
Structural priors (same magnitude as rung-3's P=8): latent dim k=2, operator order ~8.

Read DISC_LATENT only inside [DISC_LINEAR/BASE floor, ORACLE ceiling]. No tuning-to-green.
"""
import argparse, json, math, os
import torch, torch.nn as nn

# ---------------- config (identical to rung 3 unless noted) ----------------
D            = 16
K_LAT        = 2          # learned-latent dim (structural prior: symmetry is low-dim)
N_MODES      = 8
HELDOUT      = [2, 5]
RADIUS       = 4.0
SIG_LAT      = 0.08
STEPS        = 4000
AE_STEPS     = 3000       # autoencoder pretraining (DISC_LATENT only); recon-only converges ~0.03 rel
BATCH        = 512
SAMPLE_STEPS = 50
N_SAMPLE     = 2000
SEEDS        = [0, 1, 2]
W_AUG        = 1.0
W_MOVE       = 0.5
W_CYC        = 1.0
W_RECON      = 1.0        # recon anchor during joint phase (semi-freeze)
P_CYCLE      = 8
W_WD         = 1e-4
DEC_SEED     = 1234
ARMS         = ["BASE", "ORACLE", "DISC_LINEAR", "DISC_NONLIN", "DISC_LATENT"]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_MODES = [i for i in range(N_MODES) if i not in HELDOUT]


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    layers += [nn.Linear(sizes[-2], sizes[-1])]
    return nn.Sequential(*layers)


# ---------------- frozen nonlinear decoder + fixed data manifold (same as rung 3) ----------------
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
    centers = torch.stack([RADIUS * torch.cos(ang), RADIUS * torch.sin(ang)], 1)
    with torch.no_grad():
        zc = centers[torch.randint(0, N_MODES, (20000,), device=DEV)] + SIG_LAT * torch.randn(20000, 2, device=DEV)
        raw = dec(zc); mu, sd = raw.mean(0), raw.std(0) + 1e-6
    decode = lambda z: (dec(z) - mu) / sd
    with torch.no_grad():
        M_obs = decode(centers)
        zc = centers[2].repeat(4000, 1) + SIG_LAT * torch.randn(4000, 2, device=DEV)
        sig_obs = (decode(zc) - M_obs[2]).norm(dim=1).mean().item()
        pair = torch.cdist(M_obs, M_obs) + torch.eye(N_MODES, device=DEV) * 1e9
        min_pair = pair.min().item()
    return decode, centers, M_obs, sig_obs, min_pair


def rot_true():
    t = 2 * math.pi / N_MODES; c, s = math.cos(t), math.sin(t)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=DEV)


# ---------------- EqM field (same) ----------------
def eqm_loss(f, x):
    n = x.shape[0]; g = torch.rand(n, 1, device=DEV); eps = torch.randn(n, D, device=DEV)
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

class ResidualT(nn.Module):                       # rung-3 free observed-space operator
    def __init__(self, s=0.3):
        super().__init__(); self.h = mlp([D, 128, 128, D]); self.s = s
    def forward(self, x): return x + self.s * self.h(x)

class LatentT(nn.Module):                         # rung-4: dec(M . enc(x))
    def __init__(self):
        super().__init__()
        self.enc = mlp([D, 64, K_LAT])
        self.dec = mlp([K_LAT, 64, D])
        self.M = nn.Parameter(torch.eye(K_LAT) + 0.05 * torch.randn(K_LAT, K_LAT))
    def encode(self, x): return self.enc(x)
    def recon(self, x):  return self.dec(self.enc(x))
    def forward(self, x): return self.dec(self.enc(x) @ self.M.T)


# ---------------- world ----------------
DECODE, CENTERS, M_OBS, SIG_OBS, MIN_PAIR = build_world()
R_TRUE = rot_true()

def sample_latent(n):
    idx = torch.tensor(TRAIN_MODES, device=DEV)[torch.randint(0, len(TRAIN_MODES), (n,), device=DEV)]
    return CENTERS[idx] + SIG_LAT * torch.randn(n, 2, device=DEV)

def mode_of(x): return torch.cdist(x, M_OBS).argmin(1)


# ---------------- train ----------------
def train(arm, seed, steps, ae_steps):
    torch.manual_seed(seed)
    if DEV.type == "cuda": torch.cuda.manual_seed_all(seed)
    f = mlp([D + 1, 256, 256, D]).to(DEV)
    T = None; wd = 0.0; recon0 = None

    if arm == "DISC_LINEAR": T = LinearT().to(DEV); wd = W_WD
    elif arm == "DISC_NONLIN": T = ResidualT().to(DEV); wd = W_WD
    elif arm == "DISC_LATENT":
        T = LatentT().to(DEV); wd = W_WD
        # ---- AE protocol step 1: pretrain enc/dec on recon of TRAIN modes ----
        ae_opt = torch.optim.Adam(list(T.enc.parameters()) + list(T.dec.parameters()), lr=2e-3)
        for _ in range(ae_steps):
            x = DECODE(sample_latent(BATCH)); ae_opt.zero_grad()
            ((T.recon(x) - x) ** 2).mean().backward(); ae_opt.step()
        with torch.no_grad():
            x = DECODE(sample_latent(2048))
            recon0 = float((T.recon(x) - x).norm(dim=1).mean() / x.norm(dim=1).mean())
        # ---- step 2: FULLY FREEZE enc/dec; search symmetry in the fixed verified latent ----
        for p in list(T.enc.parameters()) + list(T.dec.parameters()):
            p.requires_grad_(False)

    params = list(f.parameters()) + ([p for p in T.parameters() if p.requires_grad] if T is not None else [])
    opt = torch.optim.Adam(params, lr=2e-3, weight_decay=wd)

    for _ in range(steps):
        z = sample_latent(BATCH); x = DECODE(z)
        opt.zero_grad()
        loss = eqm_loss(f, x)
        if arm == "ORACLE":
            loss = loss + W_AUG * eqm_loss(f, DECODE(z @ R_TRUE.T))
        elif arm in ("DISC_LINEAR", "DISC_NONLIN", "DISC_LATENT"):
            Tx = T(x)
            loss = loss + W_AUG * eqm_loss(f, Tx)                          # closure/aug (same recipe)
            move = (Tx - x).pow(2).sum(1).clamp_min(1e-8).sqrt()
            loss = loss + W_MOVE * (move - MIN_PAIR).pow(2).mean()         # bounded move
            xk = x
            for _ in range(P_CYCLE): xk = T(xk)
            loss = loss + W_CYC * (xk - x).pow(2).sum(1).mean()           # finite order
            # DISC_LATENT: enc/dec frozen -> latent fixed & verified; only M + field move here.
        loss.backward(); opt.step()
    return f, T, recon0


# ---------------- metrics (same as rung 3 + AE/M diagnostics) ----------------
def energy_distance(a, b):
    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).mean() - torch.cdist(b, b).mean()).item()

@torch.no_grad()
def evaluate(f, T, recon0):
    s = sample(f, N_SAMPLE); d = torch.cdist(s, M_OBS)
    near = d.argmin(1); within = d.min(1).values < 3 * SIG_OBS
    frac = torch.zeros(N_MODES, device=DEV)
    for m in range(N_MODES): frac[m] = ((near == m) & within).float().sum()
    frac = frac / N_SAMPLE
    ref = DECODE(CENTERS[torch.randint(0, N_MODES, (N_SAMPLE,), device=DEV)] + SIG_LAT * torch.randn(N_SAMPLE, 2, device=DEV))
    out = {"recall_heldout": float(frac[HELDOUT].sum()),
           "modes_covered": int((frac > 0.01).sum()),
           "mmd_energy": energy_distance(s, ref)}
    if T is not None:
        xb = DECODE(sample_latent(512)); Tx = T(xb)
        out["T_move_ratio"] = float((Tx - xb).norm(dim=1).mean() / MIN_PAIR)
        out["T_on_manifold"] = float((torch.cdist(Tx, M_OBS).min(1).values < 3 * SIG_OBS).float().mean())
        out["T_mode_change"] = float((mode_of(Tx) != mode_of(xb)).float().mean())
        sigma = {}
        for i in TRAIN_MODES:
            xi = DECODE(CENTERS[i].repeat(128, 1) + SIG_LAT * torch.randn(128, 2, device=DEV))
            sigma[i] = int(torch.bincount(mode_of(T(xi)), minlength=N_MODES).argmax().item())
        out["sigma"] = sigma
        out["hits_heldout"] = bool(any(v in HELDOUT for v in sigma.values()))
        out["consistent_shift"] = (len({(sigma[i] - i) % N_MODES for i in TRAIN_MODES}) == 1)
    if recon0 is not None:                                  # DISC_LATENT diagnostics
        out["ae_recon_rel"] = recon0
        I = torch.eye(K_LAT, device=DEV)
        Mp = T.M.clone()
        for _ in range(P_CYCLE - 1): Mp = Mp @ T.M
        out["M_offidentity"] = float((T.M - I).norm())
        out["M_order_err"] = float((Mp - I).norm())
        with torch.no_grad():
            x = DECODE(sample_latent(2048))
            out["ae_recon_rel_final"] = float((T.recon(x) - x).norm(dim=1).mean() / x.norm(dim=1).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="results_rung4.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    steps, ae_steps, seeds = (300, 300, [0]) if args.quick else (STEPS, AE_STEPS, SEEDS)

    print(f"device={DEV}  D={D}  K_lat={K_LAT}  modes={N_MODES}  heldout={HELDOUT}")
    print(f"decoder injectivity: min_pair={MIN_PAIR:.3f} sig_obs={SIG_OBS:.3f} ratio={MIN_PAIR/SIG_OBS:.1f}")
    print(f"ideal recall_heldout = {len(HELDOUT)/N_MODES:.3f}\n")

    results = {"config": {"D": D, "K_lat": K_LAT, "heldout": HELDOUT, "steps": steps,
                          "ae_steps": ae_steps, "seeds": seeds, "device": str(DEV)}, "arms": {}}
    hdr = f"{'arm':12s} {'recall@HO':>10s} {'modes/8':>8s} {'T_onman':>8s} {'hits_HO':>8s} {'cyclic':>7s} {'AE_rel':>7s} {'M-I':>6s} {'M^8-I':>7s}"
    print(hdr); print("-" * len(hdr))
    for arm in ARMS:
        runs = [evaluate(*train(arm, s, steps, ae_steps)) for s in seeds]
        agg = {}
        for k in ("recall_heldout", "modes_covered", "mmd_energy", "T_move_ratio",
                  "T_on_manifold", "T_mode_change", "ae_recon_rel", "ae_recon_rel_final",
                  "M_offidentity", "M_order_err"):
            vals = [r[k] for r in runs if k in r]
            if vals: agg[k] = sum(vals) / len(vals)
        agg["recall_sd"] = (sum((r["recall_heldout"] - agg["recall_heldout"]) ** 2 for r in runs) / len(runs)) ** 0.5
        agg["hits_heldout_any"] = any(r.get("hits_heldout") for r in runs)
        agg["consistent_shift_any"] = any(r.get("consistent_shift") for r in runs)
        agg["sigma_seed0"] = runs[0].get("sigma")
        results["arms"][arm] = agg
        def g(k, fmt="%.3f", w=7):
            return (fmt % agg[k]).rjust(w) if k in agg else "   -  ".rjust(w)
        print(f"{arm:12s} {agg['recall_heldout']:6.3f}±{agg['recall_sd']:.3f} {agg['modes_covered']:8.1f} "
              f"{g('T_on_manifold',w=8)} {str(agg['hits_heldout_any']):>8s} {str(agg['consistent_shift_any']):>7s} "
              f"{g('ae_recon_rel')} {g('M_offidentity','%.2f',6)} {g('M_order_err','%.3f',7)}")
    print(f"\nideal recall@HO = {len(HELDOUT)/N_MODES:.3f}")
    print("sigma(seed0) DISC_LATENT:", results["arms"]["DISC_LATENT"].get("sigma_seed0"))
    print("sigma(seed0) DISC_NONLIN:", results["arms"]["DISC_NONLIN"].get("sigma_seed0"))

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(results, open(args.out_json, "w"), indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
