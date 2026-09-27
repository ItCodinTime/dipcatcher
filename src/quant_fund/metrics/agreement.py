"""Inter-rater agreement: Cohen's kappa and weighted kappa.

Cohen's kappa (1960) corrects observed agreement ``p_o`` for the agreement
expected by chance ``p_e``: ``kappa = (p_o - p_e)/(1 - p_e)``.  For ordinal
labels the weighted kappa (Cohen 1968) penalises disagreements by their
distance with linear or quadratic weights.  ``kappa = 1`` is perfect agreement
and ``kappa ~ 0`` is chance agreement.

References: J. Cohen (1960), Educational and Psychological Measurement;
J. Cohen (1968), Psychological Bulletin.  Fail-closed on mismatched lengths or
too little data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]


def _confusion(y1: IntArray, y2: IntArray) -> tuple[Array, Array]:
    a = np.asarray(y1).ravel()
    b = np.asarray(y2).ravel()
    if a.size != b.size or a.size < 2:
        raise ValueError("y1 and y2 must be aligned with >= 2 observations")
    if a.dtype.kind not in "iub" or b.dtype.kind not in "iub":
        raise ValueError("labels must be finite integer or boolean categories")
    cats = np.unique(np.concatenate([a, b]))
    idx = {c: i for i, c in enumerate(cats)}
    k = cats.size
    conf = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        conf[idx[x], idx[y]] += 1.0
    return conf / conf.sum(), cats


def cohen_kappa(y1: IntArray, y2: IntArray) -> float:
    """Unweighted Cohen's kappa."""
    p, _ = _confusion(y1, y2)
    po = float(np.trace(p))
    pe = float(p.sum(axis=1) @ p.sum(axis=0))
    if pe >= 1.0:
        raise ValueError("degenerate: expected agreement is 1")
    return (po - pe) / (1.0 - pe)


def weighted_kappa(y1: IntArray, y2: IntArray, weights: str = "quadratic") -> float:
    """Weighted Cohen's kappa for ordinal labels (``linear`` or ``quadratic``)."""
    p, cats = _confusion(y1, y2)
    k = cats.size
    i = np.arange(k)
    diff = np.abs(i[:, None] - i[None, :]).astype(float)
    if weights == "quadratic":
        w = (diff / (k - 1)) ** 2 if k > 1 else diff
    elif weights == "linear":
        w = diff / (k - 1) if k > 1 else diff
    else:
        raise ValueError("weights must be 'linear' or 'quadratic'")
    expected = np.outer(p.sum(axis=1), p.sum(axis=0))
    denom = float(np.sum(w * expected))
    if denom <= 0.0:
        raise ValueError("degenerate expected disagreement")
    return float(1.0 - np.sum(w * p) / denom)
