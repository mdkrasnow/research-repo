"""Nearest-neighbor memorization + duplicate audit.

NN search is tiled GPU cosine (1 - <a,b> on L2-normalized features) with a
running top-k, so a 50K x 1.28M query x bank never materializes a full matrix
(failure mode: NN database too large for memory). Falls back to CPU when cuda
is unavailable. FAISS is optional and only used if installed AND requested.

All feature inputs are assumed L2-normalized (DINO/CLIP extractors normalize);
we renormalize defensively.
"""
from __future__ import annotations

import numpy as np
import torch


def _normalize(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def cosine_topk(
    query: np.ndarray,
    bank: np.ndarray,
    k: int = 3,
    query_labels: np.ndarray | None = None,
    bank_labels: np.ndarray | None = None,
    same_class: bool = False,
    device: str | None = None,
    query_tile: int = 2048,
    bank_tile: int = 50000,
):
    """Cosine-DISTANCE top-k (smaller = closer). Returns (dists, idx), each (Nq,k).

    same_class=True restricts each query's candidates to bank items sharing its
    label (required for class-conditional IN-1K NN). When a query's class has
    fewer than k bank items, missing slots are filled with +inf / -1.
    """
    if same_class and (query_labels is None or bank_labels is None):
        raise ValueError("same_class=True requires query_labels and bank_labels")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    q = _normalize(torch.from_numpy(np.ascontiguousarray(query)).float())
    nq = q.shape[0]
    bank_t = _normalize(torch.from_numpy(np.ascontiguousarray(bank)).float())
    bl = None if bank_labels is None else torch.from_numpy(np.asarray(bank_labels))
    ql = None if query_labels is None else np.asarray(query_labels)

    out_d = np.full((nq, k), np.inf, dtype=np.float32)
    out_i = np.full((nq, k), -1, dtype=np.int64)

    for qs in range(0, nq, query_tile):
        qe = min(qs + query_tile, nq)
        qb = q[qs:qe].to(device)
        best_d = torch.full((qe - qs, k), float("inf"), device=device)
        best_i = torch.full((qe - qs, k), -1, dtype=torch.long, device=device)
        for bs in range(0, bank_t.shape[0], bank_tile):
            be = min(bs + bank_tile, bank_t.shape[0])
            bb = bank_t[bs:be].to(device)
            sim = qb @ bb.T  # (qchunk, bchunk)
            dist = 1.0 - sim
            if same_class:
                assert ql is not None and bl is not None  # guaranteed by top-of-fn check
                qlab = torch.from_numpy(ql[qs:qe]).to(device).view(-1, 1)
                blab = bl[bs:be].to(device).view(1, -1)
                dist = dist.masked_fill(qlab != blab, float("inf"))
            kk = min(k, dist.shape[1])
            d_chunk, i_chunk = torch.topk(dist, kk, dim=1, largest=False)
            i_chunk = i_chunk + bs
            cat_d = torch.cat([best_d[:, :], d_chunk], dim=1)
            cat_i = torch.cat([best_i[:, :], i_chunk], dim=1)
            sel_d, sel = torch.topk(cat_d, k, dim=1, largest=False)
            best_d = sel_d
            best_i = torch.gather(cat_i, 1, sel)
        out_d[qs:qe] = best_d.cpu().numpy()
        out_i[qs:qe] = best_i.cpu().numpy()
    return out_d, out_i


def memorization_stats(d_train1: np.ndarray, d_val1: np.ndarray, tau_mem: float, eps: float = 1e-8):
    """d_train1/d_val1: nearest-train / nearest-val distance per generated sample."""
    d_train1 = np.asarray(d_train1, dtype=np.float64)
    d_val1 = np.asarray(d_val1, dtype=np.float64)
    ratio = d_train1 / (d_val1 + eps)
    margin = d_val1 - d_train1
    finite = np.isfinite(ratio)
    r = ratio[finite]
    return {
        "mean_d_train": float(np.mean(d_train1)),
        "median_d_train": float(np.median(d_train1)),
        "mean_d_val": float(np.mean(d_val1)),
        "median_d_val": float(np.median(d_val1)),
        "mean_ratio": float(np.mean(r)) if r.size else float("nan"),
        "median_ratio": float(np.median(r)) if r.size else float("nan"),
        "p01_ratio": float(np.quantile(r, 0.01)) if r.size else float("nan"),
        "p05_ratio": float(np.quantile(r, 0.05)) if r.size else float("nan"),
        "p10_ratio": float(np.quantile(r, 0.10)) if r.size else float("nan"),
        "frac_ratio_lt_1_0": float(np.mean(r < 1.0)) if r.size else float("nan"),
        "frac_ratio_lt_0_9": float(np.mean(r < 0.9)) if r.size else float("nan"),
        "frac_ratio_lt_0_8": float(np.mean(r < 0.8)) if r.size else float("nan"),
        "mean_margin": float(np.mean(margin)),
        "median_margin": float(np.median(margin)),
        "p95_margin": float(np.quantile(margin, 0.95)),
        "frac_train_below_tau_mem": float(np.mean(d_train1 < tau_mem)),
    }


def rank_suspicious(d_train1, d_val1, top_n=32, eps=1e-8):
    """Return dict of suspicious index lists for panels."""
    d_train1 = np.asarray(d_train1, dtype=np.float64)
    d_val1 = np.asarray(d_val1, dtype=np.float64)
    ratio = d_train1 / (d_val1 + eps)
    margin = d_val1 - d_train1
    order_dist = np.argsort(d_train1)
    order_ratio = np.argsort(ratio)
    order_margin = np.argsort(-margin)
    return {
        "lowest_train_distance": order_dist[:top_n].tolist(),
        "lowest_train_val_ratio": order_ratio[:top_n].tolist(),
        "largest_train_val_margin": order_margin[: max(1, top_n // 2)].tolist(),
    }


def calibrate_tau_dup(val_feats: np.ndarray, train_feats: np.ndarray, q=0.001, **kw):
    """tau_dup = q-quantile of val->train nearest-neighbor distances."""
    d, _ = cosine_topk(val_feats, train_feats, k=1, **kw)
    d1 = d[:, 0]
    d1 = d1[np.isfinite(d1)]
    return float(np.quantile(d1, q)) if d1.size else float("inf")


def duplicate_stats(gen_feats: np.ndarray, tau_dup: float, device: str | None = None, **kw):
    """Self-NN among generated samples; connected-components clustering."""
    n = gen_feats.shape[0]
    d, idx = cosine_topk(gen_feats, gen_feats, k=2, device=device, **kw)
    # column 0 is self (distance ~0); use column 1
    d_nearest = d[:, 1]
    j = idx[:, 1]
    near = d_nearest < tau_dup
    # union-find over near-duplicate edges
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        if near[a] and j[a] >= 0:
            ra, rb = find(a), find(int(j[a]))
            if ra != rb:
                parent[ra] = rb
    roots = {}
    for a in range(n):
        roots.setdefault(find(a), []).append(a)
    clusters = [c for c in roots.values() if len(c) > 1]
    largest = max((len(c) for c in clusters), default=1)
    d_fin = d_nearest[np.isfinite(d_nearest)]
    return {
        "mean_nearest_gen_distance": float(np.mean(d_fin)) if d_fin.size else float("nan"),
        "median_nearest_gen_distance": float(np.median(d_fin)) if d_fin.size else float("nan"),
        "p01_nearest_gen_distance": float(np.quantile(d_fin, 0.01)) if d_fin.size else float("nan"),
        "p05_nearest_gen_distance": float(np.quantile(d_fin, 0.05)) if d_fin.size else float("nan"),
        "near_duplicate_rate": float(np.mean(near)),
        "num_duplicate_clusters": int(len(clusters)),
        "largest_duplicate_cluster_size": int(largest),
        "_clusters": [c for c in sorted(clusters, key=len, reverse=True)[:16]],
    }
