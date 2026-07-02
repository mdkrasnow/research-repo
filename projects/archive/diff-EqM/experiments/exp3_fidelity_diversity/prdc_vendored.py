"""Precision / Recall / Density / Coverage (PRDC) -- vendored, pure-numpy.

Vendored (not pip-installed) to avoid a cluster network dependency at job start
and to guarantee these metrics are computed on the SAME pytorch_fid InceptionV3
2048-d features as FID/KID (feature-extractor confound control).

Algorithm: Naeem, Oh, Uh, Choi, Yoo, "Reliable Fidelity and Diversity Metrics
for Generative Models", ICML 2020 (https://github.com/clovaai/generative-evaluation-prdc,
MIT License). Reimplemented here in ~80 lines.

Definitions (k = nearest_k):
  precision : fraction of FAKE samples inside the real-manifold (union of
              k-NN balls around real samples). Fidelity.
  recall    : fraction of REAL samples inside the fake-manifold. Diversity /
              mode coverage.
  density   : softer fidelity -- mean count of real k-NN balls each fake sample
              falls into, normalized by k.
  coverage  : fraction of REAL samples whose nearest fake sample is within the
              real sample's own k-NN radius. Robust mode-coverage estimate.
"""
import numpy as np


def _pairwise_distance(a, b):
    """Euclidean distances, shape (len(a), len(b)). float64 for stability."""
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a2 = (a * a).sum(axis=1, keepdims=True)          # (Na,1)
    b2 = (b * b).sum(axis=1, keepdims=True).T         # (1,Nb)
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _kth_nn_distances(X, k):
    """Distance from each row of X to its k-th nearest neighbour within X."""
    d = _pairwise_distance(X, X)
    # exclude self (distance 0) -> the k-th NN is at sorted index k
    d_sorted = np.sort(d, axis=1)
    return d_sorted[:, k]


def compute_prdc(real_features, fake_features, nearest_k=5):
    """Returns dict with precision, recall, density, coverage.

    real_features, fake_features: (N, D) numpy arrays of Inception features.
    """
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)

    real_nn = _kth_nn_distances(real_features, nearest_k)   # (Nr,)
    fake_nn = _kth_nn_distances(fake_features, nearest_k)    # (Nf,)

    dist_rf = _pairwise_distance(real_features, fake_features)  # (Nr, Nf)

    # precision: fake inside any real ball
    precision = (dist_rf < real_nn[:, None]).any(axis=0).mean()

    # recall: real inside any fake ball
    recall = (dist_rf < fake_nn[None, :]).any(axis=1).mean()

    # density: mean (over fake) count of real balls containing it, / k
    density = (1.0 / nearest_k) * (dist_rf < real_nn[:, None]).sum(axis=0).mean()

    # coverage: fraction of real whose NEAREST fake is within real's own ball
    nearest_fake = dist_rf.min(axis=1)
    coverage = (nearest_fake < real_nn).mean()

    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
    }
