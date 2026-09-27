"""Spiegelhalter's Z test for probability-forecast calibration.

Spiegelhalter (1986) decomposes the Brier score into calibration and refinement
parts and derives an exact-variance test statistic

    Z = sum (y_i - p_i)(1 - 2 p_i) / sqrt( sum (1 - 2 p_i)^2 p_i (1 - p_i) ),

which is asymptotically standard normal under the null that the forecasts are
well calibrated.  Large ``|Z|`` indicates miscalibration.

Reference: D. Spiegelhalter (1986), "Probabilistic prediction in patient
management and clinical trials", Statistics in Medicine.  Fail-closed on
non-finite input or probabilities outside ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]


def spiegelhalter_z(prob: Array, outcome: Array) -> dict[str, float]:
    """Spiegelhalter's Z statistic and two-sided p-value."""
    p = np.asarray(prob, dtype=float).ravel()
    y = np.asarray(outcome, dtype=float).ravel()
    if p.size != y.size or p.size < 5 or not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError("prob and outcome must be finite, aligned, length >= 5")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("prob must lie in [0, 1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("outcome must be binary 0/1")
    num = float(np.sum((y - p) * (1.0 - 2.0 * p)))
    var = float(np.sum((1.0 - 2.0 * p) ** 2 * p * (1.0 - p)))
    if var <= 0.0:
        raise ValueError("degenerate variance (forecasts all 0/1)")
    z = num / np.sqrt(var)
    return {"z": float(z), "pvalue": float(2.0 * norm.sf(abs(z))), "n": float(p.size)}
