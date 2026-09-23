"""Effective number of bets / concentration warnings."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def effective_bets(weights: NDArray[np.float64], corr: NDArray[np.float64]) -> float:
    """Diversification ratio style: 1 / (w_norm^T C w_norm)."""
    w = np.asarray(weights, dtype=float)
    c = np.asarray(corr, dtype=float)
    if w.ndim != 1 or c.shape != (w.size, w.size):
        raise ValueError("corr must be square and aligned with weights")
    if not np.isfinite(w).all() or not np.isfinite(c).all():
        raise ValueError("weights and corr must be finite")
    if not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0):
        raise ValueError("corr must be symmetric with unit diagonal")
    s = np.sum(np.abs(w))
    if s == 0:
        return 0.0
    wn = np.abs(w) / s
    q = float(wn @ c @ wn)
    if q <= 0:
        return 0.0
    return float(1.0 / q)


def average_pairwise_corr(corr: NDArray[np.float64]) -> float:
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("corr must be square")
    if not np.isfinite(c).all():
        raise ValueError("corr must be finite")
    if not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0):
        raise ValueError("corr must be symmetric with unit diagonal")
    n = c.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(c[mask]))
