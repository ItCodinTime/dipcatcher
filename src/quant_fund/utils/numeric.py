"""Numerical guards."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def require_finite(x: Array, name: str = "array") -> Array:
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or inf")
    return x


def clip_positive(x: Array | float, floor: float = 1e-12) -> Array | float:
    if np.isscalar(x):
        v = float(np.asarray(x).item())
        if not np.isfinite(v) or v < floor:
            return floor
        return v
    out = np.asarray(x, dtype=float)
    out = np.where(np.isfinite(out), out, floor)
    return np.asarray(np.maximum(out, floor), dtype=np.float64)
