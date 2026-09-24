"""Expected-utility scoring under constant relative risk aversion (CRRA).

CRRA (power) utility of wealth ``w > 0`` is

    u(w) = (w^{1-gamma} - 1)/(1 - gamma)   (gamma != 1),   u(w) = log w   (gamma = 1),

with relative risk aversion ``gamma`` (Arrow-Pratt).  The certainty equivalent
of a gross-return distribution ``1 + r`` is the certain gross return with the
same expected utility; by Jensen's inequality it is below the mean for a
risk-averse investor (``gamma > 0``) and equals the mean for ``gamma = 0``.

References: J. Pratt (1964), Econometrica; K. Arrow (1965).  Fail-closed on
non-finite input or non-positive wealth.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def crra_utility(wealth: Array, gamma: float) -> Array:
    """CRRA (power) utility of positive wealth."""
    w = np.asarray(wealth, dtype=float)
    if np.any(w <= 0.0) or not np.isfinite(w).all():
        raise ValueError("wealth must be finite and positive")
    if abs(gamma - 1.0) < 1e-12:
        return np.log(w)
    return (w ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def expected_utility(returns: Array, gamma: float) -> float:
    """Expected CRRA utility of gross returns ``1 + r``."""
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 2 or not np.isfinite(r).all():
        raise ValueError("returns must be finite with >= 2 observations")
    return float(np.mean(crra_utility(1.0 + r, gamma)))


def certainty_equivalent(returns: Array, gamma: float) -> float:
    """Certainty-equivalent net return under CRRA utility."""
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 2 or not np.isfinite(r).all() or np.any(1.0 + r <= 0.0):
        raise ValueError("returns must be finite with 1 + r > 0")
    gross = 1.0 + r
    if abs(gamma - 1.0) < 1e-12:
        ce_gross = float(np.exp(np.mean(np.log(gross))))
    else:
        ce_gross = float(np.mean(gross ** (1.0 - gamma)) ** (1.0 / (1.0 - gamma)))
    return ce_gross - 1.0
