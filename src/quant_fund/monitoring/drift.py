"""Feature / prediction drift. PSI is optional, not the only metric."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def psi(reference: NDArray[np.float64], current: NDArray[np.float64], bins: int = 10) -> float:
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    r = r[np.isfinite(r)]
    c = c[np.isfinite(c)]
    if r.size < bins or c.size < bins:
        return float("nan")
    edges = np.quantile(r, np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    pr, _ = np.histogram(r, bins=edges)
    pc, _ = np.histogram(c, bins=edges)
    pr = np.clip(pr / pr.sum(), 1e-6, 1.0)
    pc = np.clip(pc / pc.sum(), 1e-6, 1.0)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def mean_shift(reference: NDArray[np.float64], current: NDArray[np.float64]) -> float:
    return float(np.nanmean(current) - np.nanmean(reference))
