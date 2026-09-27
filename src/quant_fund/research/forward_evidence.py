"""Frozen sample-size planning from pre-forward matched net-return differences."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm


def long_run_variance(values: np.ndarray, lag: int) -> float:
    x = np.asarray(values, dtype=float)
    if (
        x.ndim != 1
        or not np.isfinite(x).all()
        or type(lag) is not int
        or lag < 0
        or len(x) < max(30, 4 * (lag + 1))
    ):
        raise ValueError("HAC needs finite returns and at least max(30, 4*(lag+1)) observations")
    centered = x - x.mean()
    variance = float(centered @ centered / len(x))
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * float(centered[k:] @ centered[:-k] / len(x))
    if not np.isfinite(variance) or variance <= 1e-16:
        raise ValueError("degenerate long-run variance cannot support evidence planning")
    return variance


def evidence_plan(
    calibration: list[float],
    *,
    effect_bps: float,
    lag: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[str, Any]:
    if (
        not np.isfinite([effect_bps, alpha, power]).all()
        or effect_bps <= 0
        or not 0 < alpha < 0.5
        or not 0.5 < power < 1
    ):
        raise ValueError("effect must be positive; alpha in (0, .5), power in (.5, 1)")
    variance = long_run_variance(np.asarray(calibration), lag)
    n = int(
        np.ceil((norm.ppf(1 - alpha) + norm.ppf(power)) ** 2 * variance / (effect_bps / 1e4) ** 2)
    )
    return {
        "effect_bps": effect_bps,
        "lag": lag,
        "alpha": alpha,
        "power": power,
        "calibration_n": len(calibration),
        "long_run_variance": variance,
        "required_sessions": max(n, 30, 4 * (lag + 1)),
        "method": "one_sided_normal_approximation_with_Bartlett_HAC",
        "assumption": "future net-difference dependence resembles calibration; no power guarantee",
    }
