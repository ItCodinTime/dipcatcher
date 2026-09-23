"""Feature selection: minimum redundancy maximum relevance
(Peng, Long & Ding 2005) and related screeners.

mRMR iteratively picks the feature maximizing

    score_j = relevance_j - mean_{s in S} |corr(x_j, x_s)|

where relevance_j is |corr(x_j, y)| (FCD variant) or a univariate
F-statistic. Also provides a plain univariate screen and a simple
forward orthogonalized (Gram-Schmidt) relevance ordering.

Fail-closed: non-finite inputs, degenerate columns, k > p.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check(x: Array, y: Array) -> tuple[Array, Array]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float).ravel()
    if xx.ndim != 2 or xx.shape[0] != yy.size or xx.shape[0] < 20:
        raise ValueError("x must be (n, p) aligned with y, n >= 20")
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("non-finite input")
    if (xx.std(axis=0) <= 0).any():
        raise ValueError("degenerate column in x")
    if yy.std() <= 0:
        raise ValueError("degenerate y")
    return xx, yy


def _abs_corr(a: Array, b: Array) -> Array:
    """|corr| between columns of a and vector/b matrix b."""
    ac = a - a.mean(axis=0)
    if b.ndim == 1:
        bc = b - b.mean()
        denom = np.sqrt((ac**2).sum(axis=0) * (bc**2).sum())
        return np.abs(ac.T @ bc) / np.maximum(denom, 1e-14)
    bc = b - b.mean(axis=0)
    num = ac.T @ bc
    denom = np.sqrt((ac**2).sum(axis=0)[:, None] * (bc**2).sum(axis=0)[None, :])
    return np.abs(num) / np.maximum(denom, 1e-14)


def mrmr_select(x: Array, y: Array, k: int) -> dict[str, Array]:
    """mRMR feature selection. Returns selected indices (in order) and
    per-step scores."""
    xx, yy = _check(x, y)
    n, p = xx.shape
    if not 1 <= k <= p:
        raise ValueError("k out of range")
    relevance = _abs_corr(xx, yy)  # (p,)
    selected: list[int] = []
    scores = np.zeros(p)
    # first pick: max relevance
    j0 = int(np.argmax(relevance))
    selected.append(j0)
    scores[j0] = relevance[j0]
    for _ in range(k - 1):
        remaining = [j for j in range(p) if j not in selected]
        red = _abs_corr(xx[:, remaining], xx[:, selected]).mean(axis=1)
        mrmr = relevance[remaining] - red
        j_best = remaining[int(np.argmax(mrmr))]
        selected.append(j_best)
        scores[j_best] = float(mrmr.max())
    return {
        "selected": np.asarray(selected, dtype=np.float64),
        "scores": scores,
        "relevance": relevance,
    }


def univariate_screen(x: Array, y: Array, k: int) -> dict[str, Array]:
    """Plain univariate |corr| ranking."""
    xx, yy = _check(x, y)
    if not 1 <= k <= xx.shape[1]:
        raise ValueError("k out of range")
    rel = _abs_corr(xx, yy)
    order = np.argsort(rel)[::-1][:k]
    return {"selected": order.astype(np.float64), "relevance": rel}


def forward_orthogonalized(x: Array, y: Array, k: int) -> dict[str, Array]:
    """Forward selection: each step picks the feature whose residual
    (orthogonalized against selected) has max |corr| with y."""
    xx, yy = _check(x, y)
    n, p = xx.shape
    if not 1 <= k <= p:
        raise ValueError("k out of range")
    yc = yy - yy.mean()
    selected: list[int] = []
    xres = xx - xx.mean(axis=0)
    for _ in range(k):
        remaining = [j for j in range(p) if j not in selected]
        rel = _abs_corr(xres[:, remaining], yc)
        j = remaining[int(np.argmax(rel))]
        selected.append(j)
        # orthogonalize remaining columns against the new selection
        if len(selected) < k:
            v = xx[:, j] - xx[:, j].mean()
            for m in remaining:
                if m == j:
                    continue
                vm = xres[:, m]
                denom = float(v @ v)
                if denom > 1e-14:
                    xres[:, m] = vm - (vm @ v) / denom * v
    return {"selected": np.asarray(selected, dtype=np.float64)}
