"""Rank-association and concordance measures.

For paired observations these count concordant (``C``) and discordant (``D``)
pairs and the ties on each variable (``Tx``, ``Ty``):

- Goodman-Kruskal gamma (1954): ``(C - D)/(C + D)``.
- Somers' D (1962): ``(C - D)/(C + D + Ty)`` (``D_{y|x}``, asymmetric).
- Kendall's tau-b: ``(C - D)/sqrt((C+D+Tx)(C+D+Ty))``.

A binary-outcome concordance index (Harrell 1982) equals the AUC: the fraction
of concordant pairs among those with differing outcomes.

Fail-closed on non-finite input or too little data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _pairs(x: Array, y: Array) -> tuple[int, int, int, int]:
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size or xa.size < 3 or not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        raise ValueError("x and y must be finite, aligned, and length >= 3")
    dx = np.sign(xa[:, None] - xa[None, :])
    dy = np.sign(ya[:, None] - ya[None, :])
    iu = np.triu_indices(xa.size, k=1)
    sx, sy = dx[iu], dy[iu]
    concordant = int(np.sum(sx * sy > 0))
    discordant = int(np.sum(sx * sy < 0))
    tx = int(np.sum((sx == 0) & (sy != 0)))
    ty = int(np.sum((sy == 0) & (sx != 0)))
    return concordant, discordant, tx, ty


def goodman_kruskal_gamma(x: Array, y: Array) -> float:
    """Goodman-Kruskal gamma rank correlation."""
    c, d, _, _ = _pairs(x, y)
    if c + d == 0:
        raise ValueError("no untied pairs; gamma undefined")
    return float((c - d) / (c + d))


def somers_d(x: Array, y: Array) -> float:
    """Somers' D_{y|x} (asymmetric, y dependent on x)."""
    c, d, _, ty = _pairs(x, y)
    denom = c + d + ty
    if denom == 0:
        raise ValueError("degenerate; Somers' D undefined")
    return float((c - d) / denom)


def kendall_tau_b(x: Array, y: Array) -> float:
    """Kendall's tau-b (tie-corrected)."""
    c, d, tx, ty = _pairs(x, y)
    denom = np.sqrt((c + d + tx) * (c + d + ty))
    if denom == 0:
        raise ValueError("degenerate; tau-b undefined")
    return float((c - d) / denom)


def concordance_index(scores: Array, outcome: Array) -> float:
    """Harrell's c-index for a binary outcome (equals the ROC AUC)."""
    s = np.asarray(scores, dtype=float).ravel()
    o = np.asarray(outcome, dtype=float).ravel()
    if s.size != o.size or s.size < 2 or not np.isfinite(s).all():
        raise ValueError("scores and outcome must be finite and aligned")
    pos = s[o == 1]
    neg = s[o == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("outcome must contain both classes")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (pos.size * neg.size))
