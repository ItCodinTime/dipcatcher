"""Internal cluster-validity indices.

- **Silhouette** (Rousseeuw 1987): per-point ``s = (b - a)/max(a, b)`` with ``a``
  the mean intra-cluster distance and ``b`` the mean distance to the nearest
  other cluster; higher is better (range ``[-1, 1]``).
- **Calinski-Harabasz** (1974): ratio of between- to within-cluster dispersion,
  scaled by degrees of freedom; higher is better.
- **Davies-Bouldin** (1979): mean over clusters of the worst-case
  within-to-between spread ratio; lower is better.

Fail-closed on shape mismatch, a single cluster, or non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]


def _check(x: Array, labels: IntArray) -> tuple[Array, IntArray, IntArray]:
    arr = np.asarray(x, dtype=float)
    lab = np.asarray(labels).ravel()
    if arr.ndim != 2 or arr.shape[0] != lab.size or not np.isfinite(arr).all():
        raise ValueError("x must be (n, d) finite and aligned with labels")
    uniq = np.unique(lab)
    if uniq.size < 2:
        raise ValueError("need at least two clusters")
    return arr, lab, uniq


def silhouette_score(x: Array, labels: IntArray) -> float:
    """Mean silhouette coefficient."""
    arr, lab, uniq = _check(x, labels)
    n = arr.shape[0]
    dist = np.sqrt(((arr[:, None, :] - arr[None, :, :]) ** 2).sum(axis=2))
    sil = np.zeros(n)
    for i in range(n):
        same = lab == lab[i]
        same[i] = False
        a = dist[i, same].mean() if same.any() else 0.0
        b = np.inf
        for c in uniq:
            if c == lab[i]:
                continue
            b = min(b, dist[i, lab == c].mean())
        sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(sil.mean())


def calinski_harabasz(x: Array, labels: IntArray) -> float:
    """Calinski-Harabasz variance-ratio criterion."""
    arr, lab, uniq = _check(x, labels)
    n, k = arr.shape[0], uniq.size
    overall = arr.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in uniq:
        pts = arr[lab == c]
        centroid = pts.mean(axis=0)
        between += pts.shape[0] * float(((centroid - overall) ** 2).sum())
        within += float(((pts - centroid) ** 2).sum())
    if within <= 0.0:
        return float("inf")
    return float((between / (k - 1)) / (within / (n - k)))


def davies_bouldin(x: Array, labels: IntArray) -> float:
    """Davies-Bouldin index (lower is better)."""
    arr, lab, uniq = _check(x, labels)
    centroids = np.array([arr[lab == c].mean(axis=0) for c in uniq])
    scatter = np.array(
        [
            np.sqrt(((arr[lab == c] - centroids[i]) ** 2).sum(axis=1)).mean()
            for i, c in enumerate(uniq)
        ]
    )
    k = uniq.size
    db = 0.0
    for i in range(k):
        ratios = []
        for j in range(k):
            if i == j:
                continue
            d_ij = float(np.sqrt(((centroids[i] - centroids[j]) ** 2).sum()))
            if d_ij > 0:
                ratios.append((scatter[i] + scatter[j]) / d_ij)
        db += max(ratios) if ratios else 0.0
    return float(db / k)
