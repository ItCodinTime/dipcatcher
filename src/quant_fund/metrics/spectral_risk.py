"""Spectral risk measures (Acerbi 2002).

A spectral risk measure applies a *risk-aversion spectrum* ``phi`` to the
quantiles of the loss distribution:

    M_phi(L) = integral_0^1 phi(p) * F_L^{-1}(p) dp,

where ``phi`` is non-negative, non-decreasing, and integrates to one.  Because
``phi`` is non-decreasing it puts more weight on larger losses, which makes the
measure coherent (Acerbi 2002).  Expected shortfall is the special case of a
uniform spectrum on the worst ``1 - alpha`` tail.  The exponential spectrum of
Dowd, Cotter & Sorwar (2008),

    phi(p) = k * exp(-k (1 - p)) / (1 - exp(-k)),   k > 0,

encodes constant absolute risk aversion; larger ``k`` is more risk averse.

Losses use the positive-is-loss convention.  Fail-closed on non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _sorted_losses(losses: Array, min_obs: int = 5) -> Array:
    arr = np.asarray(losses, dtype=float).ravel()
    if arr.size < min_obs or not np.isfinite(arr).all():
        raise ValueError(f"losses must be finite with >= {min_obs} observations")
    return np.sort(arr)


def spectral_risk_measure(losses: Array, weights: Array) -> float:
    """Discrete spectral risk measure for explicit order-statistic ``weights``.

    ``weights[i]`` multiplies the i-th smallest loss; they must be
    non-negative, sum to one, and (for coherence) be non-decreasing.
    """
    ordered = _sorted_losses(losses)
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != ordered.size or not np.isfinite(w).all():
        raise ValueError("weights must be finite and aligned with losses")
    if (w < 0.0).any():
        raise ValueError("spectral weights must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("spectral weights must have positive mass")
    return float(ordered @ (w / total))


def _grid_bounds(n: int) -> tuple[Array, Array]:
    edges = np.linspace(0.0, 1.0, n + 1)
    return edges[:-1], edges[1:]


def exponential_spectral_risk(losses: Array, k: float = 5.0) -> float:
    """Exponential (CARA) spectral risk measure (Dowd, Cotter & Sorwar 2008)."""
    if k <= 0.0:
        raise ValueError("k must be positive")
    ordered = _sorted_losses(losses)
    lo, hi = _grid_bounds(ordered.size)
    # Integral of phi over each order-statistic band, phi(p)=k e^{-k(1-p)}/(1-e^{-k}).
    denom = 1.0 - np.exp(-k)
    w = (np.exp(-k * (1.0 - hi)) - np.exp(-k * (1.0 - lo))) / denom
    return spectral_risk_measure(ordered, w)


def power_spectral_risk(losses: Array, gamma: float = 2.0) -> float:
    """Power spectral risk measure, phi(p) = gamma p^{gamma-1} (gamma >= 1)."""
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1 for a non-decreasing spectrum")
    ordered = _sorted_losses(losses)
    lo, hi = _grid_bounds(ordered.size)
    w = hi**gamma - lo**gamma
    return spectral_risk_measure(ordered, w)


def expected_shortfall_srm(losses: Array, alpha: float = 0.95) -> float:
    """Expected shortfall as a spectral risk measure (uniform tail spectrum)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    ordered = _sorted_losses(losses)
    lo, hi = _grid_bounds(ordered.size)
    # Overlap of each band with the tail [alpha, 1], normalised by (1 - alpha).
    overlap = np.clip(np.minimum(hi, 1.0) - np.maximum(lo, alpha), 0.0, None)
    return spectral_risk_measure(ordered, overlap / (1.0 - alpha))
