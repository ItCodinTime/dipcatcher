"""Detrended cross-correlation analysis (DCCA) and the DCCA coefficient.

DCCA (Podobnik & Stanley 2008) measures long-range power-law cross-correlations
between two non-stationary series.  Both series are integrated (cumulative sum
of the demeaned data); for each box size ``s`` the integrated profiles are split
into overlapping windows, locally detrended by a low-order polynomial, and the
detrended cross-covariance is averaged to give ``F^2_DCCA(s)``.

The DCCA cross-correlation coefficient (Zebende 2011),

    rho_DCCA(s) = F^2_DCCA(s) / sqrt(F^2_DFA_x(s) * F^2_DFA_y(s)) in [-1, 1],

is a scale-resolved analogue of Pearson correlation.  The slope of
``log F_DCCA(s)`` vs ``log s`` is the cross-correlation exponent.

References: B. Podobnik, H. E. Stanley (2008), PRL; G. Zebende (2011),
Physica A.  Fail-closed on non-finite input or too few scales.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _profiles(x: Array, y: Array) -> tuple[Array, Array]:
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size or xa.size < 40 or not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        raise ValueError("x and y must be finite, aligned, and length >= 40")
    return np.cumsum(xa - xa.mean()), np.cumsum(ya - ya.mean())


def _box_covariance(px: Array, py: Array, s: int, order: int) -> float:
    n = px.size
    n_boxes = n - s  # overlapping windows of length s + 1
    if n_boxes <= 0:
        return float("nan")
    t = np.arange(s + 1, dtype=float)
    vander = np.vander(t, order + 1)
    covs = np.empty(n_boxes)
    for start in range(n_boxes):
        seg_x = px[start : start + s + 1]
        seg_y = py[start : start + s + 1]
        cx = np.linalg.lstsq(vander, seg_x, rcond=None)[0]
        cy = np.linalg.lstsq(vander, seg_y, rcond=None)[0]
        rx = seg_x - vander @ cx
        ry = seg_y - vander @ cy
        covs[start] = float(np.mean(rx * ry))
    return float(np.mean(covs))


def dcca(
    x: Array, y: Array, scales: Array | None = None, order: int = 1
) -> dict[str, Array | float]:
    """DCCA fluctuation, DCCA coefficient, and cross-correlation exponent."""
    px, py = _profiles(x, y)
    n = px.size
    if scales is None:
        smax = n // 4
        scales = np.unique(np.floor(np.logspace(np.log10(8), np.log10(smax), 12)).astype(int))
    s_arr = np.asarray(scales, dtype=int)
    s_arr = s_arr[(s_arr >= order + 2) & (s_arr < n)]
    if s_arr.size < 3:
        raise ValueError("need at least 3 valid scales")
    f_dcca = np.empty(s_arr.size)
    rho = np.empty(s_arr.size)
    for i, s in enumerate(s_arr):
        f2_xy = _box_covariance(px, py, int(s), order)
        f2_xx = _box_covariance(px, px, int(s), order)
        f2_yy = _box_covariance(py, py, int(s), order)
        f_dcca[i] = np.sign(f2_xy) * np.sqrt(abs(f2_xy))
        denom = np.sqrt(f2_xx * f2_yy)
        rho[i] = f2_xy / denom if denom > 0 else np.nan
    valid = np.isfinite(f_dcca) & (np.abs(f_dcca) > 0)
    if valid.sum() >= 2:
        slope = float(np.polyfit(np.log(s_arr[valid]), np.log(np.abs(f_dcca[valid])), 1)[0])
    else:
        slope = float("nan")
    return {
        "scales": s_arr.astype(float),
        "f_dcca": f_dcca,
        "rho_dcca": rho,
        "exponent": slope,
        "rho_mean": float(np.nanmean(rho)),
    }
