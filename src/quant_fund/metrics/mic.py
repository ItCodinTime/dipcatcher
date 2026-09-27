"""Maximal Information Coefficient (Reshef et al. 2011).

MIC searches over 2-D grids of the scatter of ``(x, y)`` for the binning that
maximises the normalised mutual information

    MIC = max_{n_x n_y <= B(n)}  I(X; Y) / log(min(n_x, n_y)),

where ``B(n) = n^0.6`` bounds the grid resolution.  MIC lies in ``[0, 1]``,
equals ~0 for independence and ~1 for a noiseless functional relationship of any
shape.  This implementation uses equi-frequency (quantile) binning on both axes,
a standard practical approximation to the exact dynamic-program of Reshef et
al. (2011).

Fail-closed on non-finite input or too little data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _mutual_information(counts: Array) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    outer = p.sum(axis=1, keepdims=True) @ p.sum(axis=0, keepdims=True)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / outer[mask])))


def _grid_counts(rx: Array, ry: Array, nx: int, ny: int) -> Array:
    # Equi-frequency bin edges via quantiles (robust to skew/outliers).
    xe = np.unique(np.quantile(rx, np.linspace(0.0, 1.0, nx + 1)))
    ye = np.unique(np.quantile(ry, np.linspace(0.0, 1.0, ny + 1)))
    if xe.size < 2 or ye.size < 2:
        return np.zeros((1, 1))
    counts, _, _ = np.histogram2d(rx, ry, bins=[xe, ye])
    return counts


def maximal_information_coefficient(x: Array, y: Array, alpha: float = 0.6) -> dict[str, float]:
    """Approximate MIC over equi-frequency grids; returns MIC and the best grid."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    n = xa.size
    if ya.size != n or n < 20 or not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        raise ValueError("x and y must be finite, aligned, and length >= 20")
    budget = max(4.0, n**alpha)
    best_mic = 0.0
    best_nx, best_ny = 2, 2
    for nx in range(2, int(np.sqrt(budget)) + 1):
        for ny in range(2, int(budget // nx) + 1):
            counts = _grid_counts(xa, ya, nx, ny)
            if counts.shape[0] < 2 or counts.shape[1] < 2:
                continue
            mi = _mutual_information(counts)
            norm = np.log(min(counts.shape))
            score = mi / norm if norm > 0 else 0.0
            if score > best_mic:
                best_mic = score
                best_nx, best_ny = counts.shape
    return {"mic": float(min(best_mic, 1.0)), "nx": float(best_nx), "ny": float(best_ny)}
