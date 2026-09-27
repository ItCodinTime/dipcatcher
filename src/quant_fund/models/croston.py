"""Croston's method and variants for intermittent demand / event series.

Croston (1972) forecasts sparse non-negative series (many zeros) by applying
simple exponential smoothing separately to the *non-zero sizes* ``z`` and the
*inter-arrival intervals* ``p``; the per-period rate is ``z_hat / p_hat``.

Variants:

- ``"croston"`` — the original estimator (biased high).
- ``"sba"`` — Syntetos & Boylan (2005) bias-adjusted, rate scaled by
  ``1 - alpha/2``.
- ``"tsb"`` — Teunter, Syntetos & Babai (2011); smooths the demand
  *probability* every period instead of intervals, so it adapts to
  obsolescence.

Fail-closed on negative or non-finite input and on all-zero series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_VARIANTS = ("croston", "sba", "tsb")


@dataclass(frozen=True)
class CrostonFit:
    """Fitted intermittent-demand model.  ``rate`` is the per-period forecast."""

    variant: str
    alpha: float
    beta: float
    size_level: float
    interval_level: float
    prob_level: float
    rate: float
    fitted: Array
    n_obs: int


def croston_fit(
    y: Array, alpha: float = 0.1, variant: str = "croston", beta: float | None = None
) -> CrostonFit:
    """Fit Croston / SBA / TSB.  ``beta`` (TSB probability smoothing) defaults to ``alpha``."""
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size < 3 or not np.isfinite(arr).all() or (arr < 0.0).any():
        raise ValueError("series must be finite, non-negative, with >= 3 observations")
    if not (arr > 0.0).any():
        raise ValueError("series has no non-zero demand")
    b = alpha if beta is None else float(beta)
    if not 0.0 < b < 1.0:
        raise ValueError("beta must be in (0, 1)")
    n = arr.size
    fitted = np.empty(n)

    if variant == "tsb":
        prob = float((arr > 0.0).mean())
        size = float(arr[arr > 0.0].mean())
        for t in range(n):
            fitted[t] = prob * size
            occurred = arr[t] > 0.0
            prob = b * (1.0 if occurred else 0.0) + (1.0 - b) * prob
            if occurred:
                size = alpha * arr[t] + (1.0 - alpha) * size
        rate = prob * size
        return CrostonFit(
            variant=variant,
            alpha=alpha,
            beta=b,
            size_level=size,
            interval_level=float("nan"),
            prob_level=prob,
            rate=float(rate),
            fitted=fitted,
            n_obs=n,
        )

    # Croston / SBA: SES on non-zero sizes and on intervals between them.
    nz = np.flatnonzero(arr > 0.0)
    size = float(arr[nz[0]])
    interval = float(nz[0] + 1)
    gap = 0
    ratio = size / interval
    for t in range(n):
        fitted[t] = ratio
        gap += 1
        if arr[t] > 0.0:
            size = alpha * arr[t] + (1.0 - alpha) * size
            interval = alpha * gap + (1.0 - alpha) * interval
            gap = 0
            ratio = size / interval
    correction = (1.0 - alpha / 2.0) if variant == "sba" else 1.0
    rate = correction * size / interval
    fitted = fitted * correction
    return CrostonFit(
        variant=variant,
        alpha=alpha,
        beta=b,
        size_level=size,
        interval_level=interval,
        prob_level=float("nan"),
        rate=float(rate),
        fitted=fitted,
        n_obs=n,
    )


def croston_forecast(fit: CrostonFit, h: int) -> Array:
    """Croston forecasts are constant at the estimated per-period ``rate``."""
    if h < 1:
        raise ValueError("h must be >= 1")
    return np.full(h, fit.rate)
