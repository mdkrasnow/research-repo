"""v17 MorphismPolicy — qθ(family, magnitude, composition) and the unsupervised discovery objective.

The policy knows the CANDIDATE primitive families (valid + decoy names) but NOT which are valid/useful for
a task. It learns: (1) family weights, (2) per-family magnitude distribution (mean+log_std), (3) optional
composition of depth>=2. Discovery is fully differentiable:
  - family choice per image: Gumbel-Softmax (hard, straight-through) -> grad to logits
  - magnitude: reparameterized tanh(mu + sigma*eps) -> grad through grid_sample to mu/sigma
so the energy-distance anchor loss backprops to ALL parameters with no RL.

Objective (minimize):
    L = L_anchor + a_move*L_move + a_div*L_div + a_bound*L_bound
  L_anchor : ED(features(T(visible)), anchor_ref)        -- stay ON the broad valid manifold
  L_move   : hinge(margin - mean feature move)            -- must actually transform (anti-identity)
  L_div    : -entropy(family weights)                     -- don't collapse to one family
  L_bound  : E[pre-tanh^2] beyond a soft cap              -- bounded magnitudes
Anchor-matching ALONE is solved by identity; L_move forces coverage; together they pick families that map
visible -> elsewhere in the valid manifold = the hidden morphisms. Decoys raise L_anchor (off-manifold) so
the policy is pushed off them. Ablations drop one term each (NO_ANCHOR / NO_DIVERSITY / NO_BOUNDS).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import v17_morphism_gym as G


class MorphismPolicy(nn.Module):
    def __init__(self, families, depth=1, init_logstd=-1.0):
        super().__init__()
        self.families = list(families)
        self.K = len(self.families)
        self.depth = depth
        self.logits = nn.Parameter(torch.zeros(self.K))
        self.mag_mu = nn.Parameter(torch.zeros(self.K))
        self.mag_logstd = nn.Parameter(torch.full((self.K,), init_logstd))

    def family_weights(self):
        return torch.softmax(self.logits, 0)

    def grouped_layer(self, img, gen=None):
        """One composition layer: hard-sample a family per image (~softmax(logits)), apply ONLY that family
        to its subset (1 grid_sample of work total, not K). Magnitude is reparameterized so its gradient
        flows to mag_mu/mag_logstd; the family CHOICE is a hard sample (logits trained separately by the
        EMA-reward bandit in `discover`). Returns out, fam_idx[B], pre_tanh[B], onehot[B,K]."""
        B = img.size(0)
        w = self.family_weights()
        fam_idx = torch.multinomial(w, B, replacement=True, generator=gen)                 # [B]
        eps = torch.randn(B, generator=gen, device=img.device)
        out = img.clone()
        pre = torch.zeros(B, device=img.device)
        for j, fam in enumerate(self.families):
            m = (fam_idx == j)
            if not m.any():
                continue
            pre_j = self.mag_mu[j] + torch.exp(self.mag_logstd[j]) * eps[m]
            mag_j = torch.tanh(pre_j)
            out = out.index_copy(0, m.nonzero(as_tuple=True)[0], G.apply_family(fam, img[m], mag_j))
            pre = pre.index_copy(0, m.nonzero(as_tuple=True)[0], pre_j)
        oh = F.one_hot(fam_idx, self.K).float()
        return out, fam_idx, pre, oh

    def grouped_transform(self, img, gen=None):
        out = img
        oh_acc = torch.zeros(img.size(0), self.K, device=img.device)
        pre_layers = []
        fam_last = None
        for _ in range(self.depth):
            out, fam_idx, pre, oh = self.grouped_layer(out, gen=gen)
            oh_acc = oh_acc + oh
            pre_layers.append(pre)
            fam_last = fam_idx
        return out, fam_last, torch.stack(pre_layers, 1), oh_acc

    @torch.no_grad()
    def sample_transform(self, img, gen=None):
        out, _, _, _ = self.grouped_transform(img, gen=gen)
        return out

    @torch.no_grad()
    def sample_actions(self, n, gen=None):
        """Sample n (family_idx, signed unit magnitude) actions for ONE composition layer (eval/coverage)."""
        w = self.family_weights()
        fam_idx = torch.multinomial(w, n, replacement=True, generator=gen)
        eps = torch.randn(n, generator=gen)
        mu = self.mag_mu[fam_idx]; std = torch.exp(self.mag_logstd[fam_idx])
        mag = torch.tanh(mu + std * eps)
        return fam_idx, mag


def discover(policy, visible, scorer, steps=400, lr=0.05, bs=128,
             a_move=1.0, a_div=0.3, a_bound=0.1, move_margin=0.6, use_anchor=True,
             use_diversity=True, use_bounds=True, seed=0, log_every=0, ema=0.9):
    """Train the policy. MAGNITUDES learn by reparam gradient of the anchor/move loss on the grouped
    batch; FAMILY WEIGHTS learn by a bandit: per-family reward EMA = -(group anchor-ED) (+ move bonus),
    logits pushed toward high-reward families. Cheap (1 grid_sample of work/step). Ablations drop a term.
    """
    torch.manual_seed(seed)
    K = policy.K
    # separate LRs: magnitudes (small), logits (larger) for the bandit
    opt = torch.optim.Adam([
        {"params": [policy.mag_mu, policy.mag_logstd], "lr": lr},
        {"params": [policy.logits], "lr": lr * 3},
    ])
    n = visible.size(0)
    g = torch.Generator().manual_seed(seed + 1)
    reward = torch.zeros(K)        # per-family EMA reward (higher=better)
    seen = torch.zeros(K)
    hist = []
    d = {}
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        xb = visible[idx]
        out, fam_idx, pre, oh = policy.grouped_transform(xb, gen=g)

        with torch.no_grad():
            f0 = scorer.feats(xb)
        fT = scorer.feats(out, grad=True)
        ed = G.energy_distance(fT, scorer.ref)
        move = (fT - f0).norm(dim=1).mean()

        # --- magnitude loss (reparam gradient) ---
        L = torch.zeros((), device=xb.device)
        if use_anchor:
            L = L + ed
            d["L_anchor"] = float(ed)
        L_move = F.relu(move_margin - move)
        L = L + a_move * L_move
        d["move"] = float(move); d["L_move"] = float(L_move)
        if use_bounds:
            L_bound = (F.relu(pre.abs() - 2.0) ** 2).mean()
            L = L + a_bound * L_bound
            d["L_bound"] = float(L_bound)

        # --- per-family reward for the logit bandit (no grad through grid_sample) ---
        with torch.no_grad():
            fdet = fT.detach()
            for j in range(K):
                m = (fam_idx == j)
                if m.sum() >= 8:
                    edj = float(G.energy_distance(fdet[m], scorer.ref)) if use_anchor else 0.0
                    movej = float((fdet[m] - f0[m]).norm(dim=1).mean())
                    rj = -(edj) + 0.2 * min(movej / move_margin, 2.0)
                    reward[j] = ema * reward[j] + (1 - ema) * rj if seen[j] > 0 else rj
                    seen[j] += 1
        w = policy.family_weights()
        L_logit = -(w * reward.detach()).sum()
        L = L + L_logit
        if use_diversity:
            ent = -(w * (w + 1e-9).log()).sum()
            L = L + a_div * (-ent)
            d["fam_entropy"] = float(ent)

        opt.zero_grad(); L.backward(); opt.step()
        if log_every and step % log_every == 0:
            hist.append({"step": step, "loss": float(L), **d})
    out_diag = {"final": d, "hist": hist,
                "family_weights": {f: float(w) for f, w in zip(policy.families, policy.family_weights())},
                "mag_mu": {f: float(m) for f, m in zip(policy.families, policy.mag_mu.detach())},
                "reward": {f: float(r) for f, r in zip(policy.families, reward)}}
    return out_diag


# ---------------------------------------------------------------------------
# Reference (non-learned) policies for controls.

class FixedPolicy:
    """Applies a fixed set of families with random magnitudes. Used for ORACLE / RANDOM_VALID /
    RANDOM_WITH_DECOYS / identity controls."""
    def __init__(self, families, mag_lo=0.4, mag_hi=1.0, depth=1, identity=False):
        self.families = list(families)
        self.mag_lo, self.mag_hi = mag_lo, mag_hi
        self.depth = depth
        self.identity = identity

    @torch.no_grad()
    def sample_actions(self, n, gen=None):
        if not self.families:
            return torch.zeros(n, dtype=torch.long), torch.zeros(n)
        fam_idx = torch.randint(0, len(self.families), (n,), generator=gen)
        mag = (torch.rand(n, generator=gen) * (self.mag_hi - self.mag_lo) + self.mag_lo)
        mag = mag * (torch.randint(0, 2, (n,), generator=gen) * 2 - 1)
        return fam_idx, mag

    @torch.no_grad()
    def sample_transform(self, img, gen=None):
        if self.identity or not self.families:
            return img
        out = img
        for _ in range(self.depth):
            B = out.size(0)
            fam_idx = torch.randint(0, len(self.families), (B,), generator=gen)
            mag = (torch.rand(B, generator=gen) * (self.mag_hi - self.mag_lo) + self.mag_lo)
            mag = mag * (torch.randint(0, 2, (B,), generator=gen) * 2 - 1)
            res = out.clone()
            for j, fam in enumerate(self.families):
                m = (fam_idx == j)
                if m.any():
                    res[m] = G.apply_family(fam, out[m], mag[m])
            out = res
        return out
