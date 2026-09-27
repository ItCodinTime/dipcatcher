"""Ledoit-Wolf (2008) test for the difference of two Sharpe ratios.

Comparing two strategies' Sharpe ratios requires accounting for the estimation
uncertainty of both the mean and the variance, and for autocorrelation.  Ledoit
& Wolf (2008) derive a HAC (Newey-West) standard error for the Sharpe
difference via the delta method on the moment vector
``theta = (mu1, mu2, gamma1, gamma2)`` with ``gamma_i = E[r_i^2]`` and
``SR_i = mu_i / sqrt(gamma_i - mu_i^2)``.  The studentised statistic is
asymptotically standard normal.

Reference: O. Ledoit, M. Wolf (2008), "Robust performance hypothesis testing
with the Sharpe ratio", Journal of Empirical Finance.  Fail-closed on
non-finite input or degenerate variance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]


def _hac_cov(v: Array, lag: int) -> Array:
    t = v.shape[0]
    vc = v - v.mean(axis=0)
    omega = vc.T @ vc / t
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        g = vc[j:].T @ vc[:-j] / t
        omega += w * (g + g.T)
    return np.asarray(omega, dtype=float)


def sharpe_difference_test(
    r1: Array, r2: Array, lag: int | None = None
) -> dict[str, float]:
    """Ledoit-Wolf HAC test of ``H0: SR1 = SR2`` for paired return series."""
    a = np.asarray(r1, dtype=float).ravel()
    b = np.asarray(r2, dtype=float).ravel()
    if (
        a.size != b.size
        or a.size < 30
        or not (np.isfinite(a).all() and np.isfinite(b).all())
    ):
        raise ValueError("r1 and r2 must be finite, aligned, length >= 30")
    t = a.size
    if lag is not None and (
        isinstance(lag, bool)
        or not isinstance(lag, (int, np.integer))
        or not 0 <= lag < t
    ):
        raise ValueError("lag must be an integer from 0 to n-1")
    mu1, mu2 = float(a.mean()), float(b.mean())
    g1, g2 = float((a**2).mean()), float((b**2).mean())
    s1 = g1 - mu1**2
    s2 = g2 - mu2**2
    if not np.isfinite([g1, g2, s1, s2]).all() or s1 <= 0.0 or s2 <= 0.0:
        raise ValueError("degenerate variance")
    sr1, sr2 = mu1 / np.sqrt(s1), mu2 / np.sqrt(s2)
    diff = float(sr1 - sr2)
    # gradient of (SR1 - SR2) wrt (mu1, mu2, gamma1, gamma2)
    grad = np.array(
        [
            g1 / s1**1.5,
            -g2 / s2**1.5,
            -0.5 * mu1 / s1**1.5,
            0.5 * mu2 / s2**1.5,
        ]
    )
    v = np.column_stack([a, b, a**2, b**2])
    lag = int(np.floor(4.0 * (t / 100.0) ** (2.0 / 9.0))) if lag is None else lag
    omega = _hac_cov(v, lag)
    se2 = float(grad @ omega @ grad / t)
    if not np.isfinite(se2) or se2 <= 0.0:
        raise ValueError("degenerate standard error")
    se = float(np.sqrt(se2))
    z = diff / se
    return {
        "sr1": float(sr1),
        "sr2": float(sr2),
        "diff": diff,
        "se": se,
        "z": float(z),
        "pvalue": float(2.0 * norm.sf(abs(z))),
    }
