"""Feature / prediction drift. PSI is optional, not the only metric."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def drift_report(
    reference: NDArray[np.float64],
    current: NDArray[np.float64],
    *,
    psi_threshold: float = 0.25,
    mean_shift_threshold: float | None = None,
    bins: int = 10,
) -> dict[str, float | bool | str]:
    """Return an auditable drift decision with explicit sample sufficiency.

    PSI is only meaningful when both windows support the requested bins. A
    short or empty window therefore produces ``insufficient_data`` rather
    than a false all-clear. Mean shift is optional and remains a diagnostic.
    """
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    n_reference = int(np.isfinite(r).sum())
    n_current = int(np.isfinite(c).sum())
    value = psi(r, c, bins=bins)
    shift = mean_shift(r, c)
    sufficient = bool(np.isfinite(value))
    psi_alert = bool(sufficient and value >= float(psi_threshold))
    mean_alert = bool(
        mean_shift_threshold is not None
        and np.isfinite(shift)
        and abs(shift) >= float(mean_shift_threshold)
    )
    return {
        "status": "alert"
        if psi_alert or mean_alert
        else ("ok" if sufficient else "insufficient_data"),
        "psi": float(value),
        "mean_shift": float(shift),
        "psi_threshold": float(psi_threshold),
        "n_reference": float(n_reference),
        "n_current": float(n_current),
        "sufficient_data": sufficient,
        "alert": psi_alert or mean_alert,
    }


def psi(reference: NDArray[np.float64], current: NDArray[np.float64], bins: int = 10) -> float:
    if int(bins) < 2:
        raise ValueError("bins must be >= 2")
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    r = r[np.isfinite(r)]
    c = c[np.isfinite(c)]
    if r.size < bins or c.size < bins:
        return float("nan")
    edges = np.quantile(r, np.linspace(0, 1, bins + 1))
    # Keep current-window outliers in the outer bins instead of dropping them.
    edges[0] = min(edges[0], float(np.min(c)))
    edges[-1] = max(edges[-1], float(np.max(c)))
    if np.any(np.diff(edges) <= 0.0):
        edges = np.linspace(
            float(min(np.min(r), np.min(c))), float(max(np.max(r), np.max(c))), bins + 1
        )
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    pr, _ = np.histogram(r, bins=edges)
    pc, _ = np.histogram(c, bins=edges)
    pr = np.clip(pr / pr.sum(), 1e-6, 1.0)
    pc = np.clip(pc / pc.sum(), 1e-6, 1.0)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def mean_shift(reference: NDArray[np.float64], current: NDArray[np.float64]) -> float:
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    r = r[np.isfinite(r)]
    c = c[np.isfinite(c)]
    if r.size == 0 or c.size == 0:
        return float("nan")
    return float(np.mean(c) - np.mean(r))
