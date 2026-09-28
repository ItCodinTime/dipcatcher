"""Threat-model perturbations.

Perturbations are pure functions of an array the caller already holds. They
do not read a market feed and they do not submit orders.

Units
-----
Volatility-scaled path noise is ``eta`` with ``||eta||`` bounded, applied as
``returns + vol * eta`` or, on prices, as the same shock to log increments.
Missing bars, stale prints, and spikes are discrete corruptions. Timing
jitter rolls a series by an integer number of bars. Cost shocks are an
additive change to a cost rate. Regime shift is not a path edit: it is a
Wasserstein ball on an outcome law and is implemented in ``dro``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

THREAT_NAMES: tuple[str, ...] = (
    "vol_scaled_path",
    "missing_bars",
    "stale_prints",
    "spikes",
    "timing_jitter",
    "cost_shock",
    "wasserstein_regime",
)


def as_vector(values: FloatArray, *, name: str) -> FloatArray:
    """Return a finite 1-d float copy."""
    array = np.array(values, dtype=float, copy=True).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def project(delta: FloatArray, radius: float, norm: str) -> FloatArray:
    """Project ``delta`` onto the origin-centered ``norm`` ball."""
    if norm not in {"l2", "linf"}:
        raise ValueError("norm must be 'l2' or 'linf'")
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and non-negative")
    vector = np.asarray(delta, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("delta must be finite")
    if radius == 0.0:
        return np.zeros_like(vector)
    if norm == "linf":
        return np.asarray(np.clip(vector, -radius, radius), dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= radius or length == 0.0:
        return vector.copy()
    return vector * (radius / length)


def ball_norm(delta: FloatArray, norm: str) -> float:
    """``l2`` or ``linf`` norm of a vector."""
    vector = np.asarray(delta, dtype=float).reshape(-1)
    if norm == "l2":
        return float(np.linalg.norm(vector))
    if norm == "linf":
        return float(np.max(np.abs(vector))) if vector.size else 0.0
    raise ValueError("norm must be 'l2' or 'linf'")


def apply_vol_scaled(
    sample: FloatArray, eta: FloatArray, scale: FloatArray | None = None
) -> FloatArray:
    """Add a volatility-scaled perturbation to a return or feature path.

    ``scale`` defaults to ones, so ``eta`` is then in the path's own units.
    """
    base = as_vector(sample, name="sample")
    shock = np.asarray(eta, dtype=float).reshape(-1)
    if shock.shape != base.shape or not np.all(np.isfinite(shock)):
        raise ValueError("eta must be finite and aligned with sample")
    if scale is None:
        vol = np.ones_like(base)
    else:
        vol = np.asarray(scale, dtype=float).reshape(-1)
        if vol.shape != base.shape or not np.all(np.isfinite(vol)) or np.any(vol < 0.0):
            raise ValueError("scale must be finite, non-negative, and aligned")
    return np.asarray(base + vol * shock, dtype=float)


def apply_vol_scaled_prices(prices: FloatArray, eta: FloatArray, vol: FloatArray) -> FloatArray:
    """Apply ``eta`` to log-price increments and rebuild a positive price path.

    ``prices`` has length T. ``eta`` and ``vol`` align with the T-1 increments.
    Increment ``t`` (the move from ``t`` to ``t+1``) changes by ``vol[t] * eta[t]``.
    """
    path = as_vector(prices, name="prices")
    if np.any(path <= 0.0):
        raise ValueError("prices must be positive")
    shock = np.asarray(eta, dtype=float).reshape(-1)
    scale = np.asarray(vol, dtype=float).reshape(-1)
    increments = np.diff(np.log(path))
    if shock.shape != increments.shape or scale.shape != increments.shape:
        raise ValueError("eta and vol must align with price increments")
    if not np.all(np.isfinite(shock)) or not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
        raise ValueError("eta and vol must be finite and vol non-negative")
    out = np.empty_like(path)
    out[0] = path[0]
    log_level = math.log(float(path[0]))
    for index, (increment, eta_i, vol_i) in enumerate(zip(increments, shock, scale, strict=True)):
        log_level += float(increment + vol_i * eta_i)
        out[index + 1] = math.exp(log_level)
    return out


def causal_increment_vol(prices: FloatArray, *, floor: float = 1e-8) -> FloatArray:
    """Lagged standard deviation of log increments, aligned to increments.

    ``vol[t]`` uses increments ``[:t]`` only, so the scale on the current
    increment does not depend on that increment. The first scale is ``floor``
    because there is no history. This is a diagnostic scale, not a forecast.
    """
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be finite and positive")
    path = as_vector(prices, name="prices")
    if path.size < 2 or np.any(path <= 0.0):
        raise ValueError("prices must contain at least two positive observations")
    increments = np.diff(np.log(path))
    vol = np.empty(increments.shape, dtype=float)
    vol[0] = floor
    for index in range(1, increments.size):
        history = increments[:index]
        estimate = float(np.std(history, ddof=1)) if history.size >= 2 else floor
        vol[index] = estimate if math.isfinite(estimate) and estimate > 0.0 else floor
    return vol


def apply_missing(sample: FloatArray, indices: list[int]) -> FloatArray:
    """Zero selected coordinates. A missing bar is recorded as no move.

    The fill is a modeling choice and is disclosed by the caller. It is not
    a claim about how a venue prints empty intervals.
    """
    out = as_vector(sample, name="sample").copy()
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("missing-bar indices must be integers")
        if index < 0 or index >= out.size:
            raise IndexError("missing-bar index out of range")
        out[index] = 0.0
    return out


def apply_stale(sample: FloatArray, start: int, length: int) -> FloatArray:
    """Repeat the previous print for ``length`` bars starting at ``start``.

    A window at the first bar is filled with 0 because no previous print
    exists. ``length`` is the number of replaced bars.
    """
    out = as_vector(sample, name="sample").copy()
    if isinstance(start, bool) or isinstance(length, bool):
        raise TypeError("start and length must be integers")
    if not isinstance(start, int) or not isinstance(length, int):
        raise TypeError("start and length must be integers")
    if start < 0 or start >= out.size or length < 1:
        raise ValueError("stale window is outside the sample")
    end = min(out.size, start + length)
    fill = 0.0 if start == 0 else float(out[start - 1])
    out[start:end] = fill
    return out


def apply_spike(sample: FloatArray, index: int, magnitude: float, scale: float = 1.0) -> FloatArray:
    """Add ``magnitude * scale`` to one coordinate."""
    out = as_vector(sample, name="sample").copy()
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= out.size:
        raise ValueError("spike index is outside the sample")
    if not math.isfinite(magnitude) or not math.isfinite(scale) or scale < 0.0:
        raise ValueError("spike magnitude and scale must be finite; scale non-negative")
    out[index] = float(out[index] + magnitude * scale)
    return out


def apply_jitter(sample: FloatArray, shift: int) -> FloatArray:
    """Roll a series by ``shift`` bars.

    Positive ``shift`` delays observations (a late print). Negative ``shift``
    advances them. The roll is circular so the threat stays inside a fixed
    window; it is a diagnostic on that window, not a causal filter.
    """
    out = as_vector(sample, name="sample")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise TypeError("shift must be an integer")
    if out.size == 0:
        return out
    return np.asarray(np.roll(out, shift), dtype=float)


def path_turnover(positions: FloatArray, initial: float = 0.0) -> float:
    """Sum of absolute position changes, starting from ``initial``."""
    held = np.asarray(positions, dtype=float).reshape(-1)
    if held.size == 0 or not np.all(np.isfinite(held)) or not math.isfinite(initial):
        raise ValueError("positions must be non-empty and finite")
    changes = np.diff(held, prepend=initial)
    return float(np.sum(np.abs(changes)))


def path_gross(positions: FloatArray, returns: FloatArray) -> float:
    """Sum of ``position * return`` on one aligned path. Not a live result."""
    held = np.asarray(positions, dtype=float).reshape(-1)
    rets = np.asarray(returns, dtype=float).reshape(-1)
    if held.shape != rets.shape or not np.all(np.isfinite(held)) or not np.all(np.isfinite(rets)):
        raise ValueError("positions and returns must be finite and aligned")
    return float(np.sum(held * rets))


def net_excess(
    positions: FloatArray, returns: FloatArray, cost: float, initial: float = 0.0
) -> float:
    """Gross sum minus ``cost`` times turnover on one simulated path."""
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError("cost must be finite and non-negative")
    return path_gross(positions, returns) - cost * path_turnover(positions, initial=initial)
