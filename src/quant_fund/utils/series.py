"""Shared series and panel guards.

Callers keep their own minimum-length defaults. The checks themselves are
one implementation so a length or finiteness rule cannot drift between modules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def finite_series(values: Array, min_length: int) -> Array:
    """Require a finite 1-d series of at least ``min_length`` observations."""
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size < min_length or not np.all(np.isfinite(series)):
        raise ValueError(f"series must be finite with length >= {min_length}")
    return series


def finite_nonconstant_series(values: Array, min_length: int) -> Array:
    """Like ``finite_series``, and reject a zero-variance series."""
    series = finite_series(values, min_length)
    if series.std() == 0:
        raise ValueError("degenerate (constant) series")
    return series


def finite_observations(values: Array, name: str = "x", *, min_obs: int) -> Array:
    """Drop non-finite points, then require ``min_obs`` observations remain."""
    series = np.asarray(values, dtype=float).reshape(-1)
    series = series[np.isfinite(series)]
    if series.size < min_obs:
        raise ValueError(f"{name} must contain at least {min_obs} finite observations")
    return series


def finite_1d(values: Array) -> Array:
    """Return the finite entries of a 1-d view. Empty input stays empty."""
    series = np.asarray(values, dtype=float).reshape(-1)
    return series[np.isfinite(series)]


def as_named_1d(name: str, values: Array) -> Array:
    """Reject rank > 1. A row vector is flattened; a matrix is an error."""
    array = np.asarray(values, dtype=float)
    if array.ndim > 1:
        raise ValueError(f"{name} must be 1d")
    return array.reshape(-1)


def require_same_length(*named: tuple[str, Array]) -> None:
    """Fail closed when named 1-d arrays disagree on length."""
    lengths = {name: array.shape[0] for name, array in named}
    if len(set(lengths.values())) > 1:
        parts = ", ".join(f"{key}={value}" for key, value in lengths.items())
        raise ValueError(f"length mismatch: {parts}")


def require_panel(values: Array, *, min_n: int, min_t: int) -> Array:
    """Require a finite ``(T, N)`` panel. Defaults stay at the call site."""
    panel = np.asarray(values, dtype=float)
    if panel.ndim != 2 or panel.shape[0] < min_t or panel.shape[1] < min_n:
        raise ValueError(f"panel must be (T >= {min_t}, N >= {min_n})")
    if not np.all(np.isfinite(panel)):
        raise ValueError("panel must be finite")
    return panel
