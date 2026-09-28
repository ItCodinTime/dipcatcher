"""Threshold-weighted CRPS — proper scoring with region emphasis.

Gneiting & Ranjan (2011, "Comparing density forecasts using threshold- and
quantile-weighted scoring rules"): for a weight function w(z) >= 0,

    twCRPS(F, y) = integral (F(z) - 1{y <= z})^2 w(z) dz.

With w = 1 this is plain CRPS. With w(z) = 1{z >= tau} only tail errors
beyond tau count — the recommended way to emphasize tails while keeping the
score proper. The ensemble-CDF integral is computed exactly on the sorted
support (F piecewise-constant between grid points of ens ∪ {y}); arbitrary
smooth weights are integrated per-segment by Gauss-Legendre quadrature.

Fail-closed: empty/non-finite ensembles, non-finite outcomes, negative
weights.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_LEG_NODES, _LEG_WEIGHTS = np.polynomial.legendre.leggauss(48)


def _check_ens(ens: Array, y: float) -> Array:
    e = np.asarray(ens, dtype=float).ravel()
    if e.size < 2 or not np.isfinite(e).all():
        raise ValueError("ens must be a finite array with >= 2 members")
    if not np.isfinite(y):
        raise ValueError("y must be finite")
    return np.sort(e)


def crps_ensemble(ens: Array, y: float) -> float:
    """Standard ensemble CRPS: E|X - y| - 0.5 E|X - X'|."""
    e = _check_ens(ens, y)
    m = e.size
    term1 = np.mean(np.abs(e - y))
    term2 = float(np.sum(np.abs(e[:, None] - e[None, :]))) / (2.0 * m * m)
    return float(term1 - term2)


def _twcrps(
    e: Array,
    y: float,
    weight: Callable[[Array], Array] | None,
    tau: float | None,
) -> float:
    grid = np.unique(np.concatenate([e, [y]]))
    seg_lo, seg_hi = grid[:-1], grid[1:]
    mid = 0.5 * (seg_lo + seg_hi)
    if tau is not None:
        seg_w = np.clip(seg_hi - np.maximum(seg_lo, tau), 0.0, None)
    elif weight is None:
        seg_w = seg_hi - seg_lo
    else:
        half = 0.5 * (seg_hi - seg_lo)
        z = mid[:, None] + half[:, None] * _LEG_NODES[None, :]
        wvals = np.asarray(weight(z), dtype=float)
        if wvals.shape != z.shape or not np.isfinite(wvals).all():
            raise ValueError("weight must return finite values on the data range")
        if (wvals < -1e-12).any():
            raise ValueError("weight must be non-negative")
        seg_w = (wvals * half[:, None]) @ _LEG_WEIGHTS
    cdf = np.searchsorted(e, mid, side="right") / e.size
    ind = (y <= mid).astype(float)
    return float(np.sum((cdf - ind) ** 2 * seg_w))


def twcrps_ensemble(
    ens: Array,
    y: float,
    weight: Callable[[Array], Array] | None = None,
) -> float:
    """Threshold-weighted ensemble CRPS with arbitrary smooth ``weight``.

    ``weight(z)`` maps observation space to non-negative weights; pass None
    for plain CRPS (exact, equals ``crps_ensemble``).
    """
    e = _check_ens(ens, y)
    return _twcrps(e, float(y), weight, None)


def twcrps_tail(ens: Array, y: float, tau: float) -> float:
    """Tail-weighted CRPS with indicator weight w(z) = 1{z >= tau} (exact)."""
    e = _check_ens(ens, y)
    if not np.isfinite(tau):
        raise ValueError("tau must be finite")
    return _twcrps(e, float(y), None, float(tau))
