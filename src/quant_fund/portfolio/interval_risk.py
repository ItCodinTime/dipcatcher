"""Map conformal prediction sets to executable name caps. No P&L, no Sharpe.

See docs/decisions/014-interval-position-caps.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

DEFAULT_MAX_WEIGHT = 0.02
DEFAULT_WIDTH_REF = 0.10
DEFAULT_DOWNSIDE_REF = 0.05


def _as_aligned(lo: Array | float, hi: Array | float) -> tuple[Array, Array]:
    a = np.asarray(lo, dtype=float)
    b = np.asarray(hi, dtype=float)
    if a.shape != b.shape:
        raise ValueError("lo and hi must have the same shape")
    return a, b


def equal_weight_per_date(dates: NDArray[Any] | list[object]) -> Array:
    """Cross-sectional equal weight ``1/n_names`` on each date.

    Every name on a date gets the same weight. This is not a median cap.
    """
    keys = np.asarray([str(d) for d in dates], dtype=object)
    n = int(keys.size)
    if n == 0:
        return np.asarray([], dtype=float)
    _uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return (1.0 / counts.astype(np.float64))[inv]


def interval_width(lo: Array | float, hi: Array | float) -> Array:
    """Elementwise set width ``hi - lo`` (may be negative if inverted)."""
    a, b = _as_aligned(lo, hi)
    return b - a


def downside(lo: Array | float) -> Array:
    """How negative the lower endpoint is: ``max(0, -lo)``."""
    return np.maximum(0.0, -np.asarray(lo, dtype=float))


def interval_refs(
    lo: Array | float,
    hi: Array | float,
    *,
    multiple: float = 2.0,
) -> tuple[float, float]:
    """Calibrate width/downside refs from a set of intervals (typically calibration).

    Refs are ``multiple * median`` so a typical name keeps most of the box and
    only unusually wide or left-tailed sets shrink hard.
    """
    if multiple <= 0.0:
        raise ValueError("multiple must be > 0")
    lo_a, hi_a = _as_aligned(lo, hi)
    valid = np.isfinite(lo_a) & np.isfinite(hi_a) & (hi_a >= lo_a)
    if int(valid.sum()) == 0:
        return DEFAULT_WIDTH_REF, DEFAULT_DOWNSIDE_REF
    width = np.maximum(interval_width(lo_a[valid], hi_a[valid]), 0.0)
    down = downside(lo_a[valid])
    wr = float(multiple * np.nanmedian(width))
    dr = float(multiple * np.nanmedian(down))
    return max(wr, 1e-8), max(dr, 1e-8)


def cap_from_interval(
    lo: Array | float,
    hi: Array | float,
    *,
    max_weight: float,
    width_ref: float,
    downside_ref: float,
) -> Array:
    """Soft name caps in ``[0, max_weight]`` from a conformal interval.

    cap = max_weight * (width_ref / (width_ref + width))
        * (downside_ref / (downside_ref + downside))

    Wider sets or a more negative lower tail strictly shrink the cap.
    Missing, non-finite, or inverted intervals fail closed to 0.
    Deterministic; no randomness.
    """
    if max_weight < 0.0:
        raise ValueError("max_weight must be >= 0")
    if width_ref <= 0.0 or downside_ref <= 0.0:
        raise ValueError("width_ref and downside_ref must be > 0")

    lo_a, hi_a = _as_aligned(lo, hi)
    width = np.maximum(interval_width(lo_a, hi_a), 0.0)
    down = downside(lo_a)
    valid = np.isfinite(lo_a) & np.isfinite(hi_a) & (hi_a >= lo_a)

    shrink_width = width_ref / (width_ref + width)
    shrink_down = downside_ref / (downside_ref + down)
    raw = max_weight * shrink_width * shrink_down
    capped = np.clip(raw, 0.0, max_weight)
    return np.where(valid, capped, 0.0)


def apply_interval_caps(
    weights: Array | float,
    lo: Array | float,
    hi: Array | float,
    *,
    max_weight: float,
    width_ref: float,
    downside_ref: float,
) -> tuple[Array, Array]:
    """Clip ``|w|`` to the interval cap, preserving sign.

    Returns ``(capped_weights, caps)``. Does not renormalize the book.
    Non-finite weights fail closed to 0.
    """
    caps = cap_from_interval(
        lo, hi, max_weight=max_weight, width_ref=width_ref, downside_ref=downside_ref
    )
    w = np.asarray(weights, dtype=float)
    if w.shape != caps.shape:
        raise ValueError("weights must have the same shape as lo and hi")
    finite_w = np.isfinite(w)
    signed = np.sign(w)
    clipped = signed * np.minimum(np.abs(w), caps)
    capped = np.where(finite_w, clipped, 0.0)
    return capped, caps


def bench_interval_caps(
    lo: Array | float,
    hi: Array | float,
    weights: Array | float,
    *,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    width_ref: float = DEFAULT_WIDTH_REF,
    downside_ref: float = DEFAULT_DOWNSIDE_REF,
) -> dict[str, float]:
    """Book-level interval-cap diagnostics. Coverage-set geometry only; no Sharpe."""
    capped, caps = apply_interval_caps(
        weights,
        lo,
        hi,
        max_weight=max_weight,
        width_ref=width_ref,
        downside_ref=downside_ref,
    )
    w = np.asarray(weights, dtype=float)
    widths = interval_width(lo, hi)
    return {
        "mean_cap": float(np.mean(caps)),
        "frac_binding": float(np.mean(np.abs(w) > caps)),
        "mean_width": float(np.nanmean(widths)),
        "n": float(caps.size),
        "gross_after": float(np.sum(np.abs(capped))),
    }
