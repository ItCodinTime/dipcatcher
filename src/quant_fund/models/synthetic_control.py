"""Synthetic control method (Abadie & Gardeazabal 2003; Abadie, Diamond &
Hainmueller 2010).

``synthetic_control`` fits simplex weights w >= 0, sum w = 1 minimizing the
pre-treatment fit error

    min_w || y1_pre - Y0_pre w ||^2,        w in the probability simplex,

via fast projected-gradient descent with the exact Duchi et al. (2008)
simplex projection. ``placebo_test`` runs the in-space placebo distribution:
each donor is treated in turn and the treated unit's post/pre RMSPE ratio is
ranked against the donors'.

Returns weights, synthetic path, gap series, and pre/post RMSPE. Fail-closed:
non-finite input, degenerate donor panels, T0 out of range.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check_panel(y0: Array, y1: Array, t0: int) -> tuple[Array, Array, int]:
    y0 = np.asarray(y0, dtype=float)
    y1 = np.asarray(y1, dtype=float).ravel()
    if y0.ndim != 2:
        raise ValueError("Y0 must be (T, J) donor panel")
    t_len, j_n = y0.shape
    if y1.size != t_len:
        raise ValueError("y1 length must match Y0 rows")
    if not (2 <= t0 < t_len):
        raise ValueError("T0 must satisfy 2 <= T0 < T")
    if j_n < 2:
        raise ValueError("need at least 2 donors")
    if not np.isfinite(y0).all() or not np.isfinite(y1).all():
        raise ValueError("non-finite input")
    if (y0[:t0].std(axis=0) <= 1e-14).all():
        raise ValueError("degenerate pre-period donor panel")
    return y0, y1, t_len


def simplex_project(v: Array) -> Array:
    """Exact Euclidean projection onto the probability simplex."""
    x = np.asarray(v, dtype=float).ravel()
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("v must be a finite non-empty array")
    u = np.sort(x)[::-1]
    css = np.cumsum(u)
    rho_idx = np.nonzero(u - (css - 1.0) / np.arange(1, x.size + 1) > 0)[0]
    rho = int(rho_idx[-1]) if rho_idx.size else 0
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.asarray(np.maximum(x - theta, 0.0))


def synthetic_control(
    y0: Array,
    y1: Array,
    t0: int,
    n_iter: int = 3000,
    step: float | None = None,
) -> dict[str, Array | float]:
    """Fit simplex-constrained synthetic control weights."""
    y0, y1, _ = _check_panel(y0, y1, t0)
    if n_iter < 10:
        raise ValueError("n_iter must be >= 10")
    a = y0[:t0]
    b = y1[:t0]
    g = a.T @ a
    c = -2.0 * (a.T @ b)
    lips = float(np.linalg.norm(g, 2)) + 1e-12
    lr = float(step) if step is not None else 1.0 / lips
    if lr <= 0 or not np.isfinite(lr):
        raise ValueError("step must be positive and finite")
    w = np.full(a.shape[1], 1.0 / a.shape[1])
    for _ in range(n_iter):
        w = simplex_project(w - lr * (g @ w * 2.0 + c) * 0.5)
    synth = y0 @ w
    gap = y1 - synth
    rmspe_pre = float(np.sqrt(np.mean(gap[:t0] ** 2)))
    rmspe_post = float(np.sqrt(np.mean(gap[t0:] ** 2)))
    return {
        "w": w,
        "synthetic": synth,
        "gap": gap,
        "rmspe_pre": rmspe_pre,
        "rmspe_post": rmspe_post,
        "rmspe_ratio": float(rmspe_post / max(rmspe_pre, 1e-12)),
        "att_post": float(np.mean(gap[t0:])),
    }


def placebo_test(
    y_all: Array,
    treated_idx: int,
    t0: int,
    n_iter: int = 1500,
) -> dict[str, Array | float]:
    """In-space placebo distribution over all units of ``y_all`` (T, J+1)."""
    panel = np.asarray(y_all, dtype=float)
    if panel.ndim != 2 or panel.shape[1] < 3:
        raise ValueError("y_all must be (T, J+1) with >= 2 donors")
    if not (0 <= treated_idx < panel.shape[1]):
        raise ValueError("treated_idx out of range")
    t_len, n_units = panel.shape
    if not (2 <= t0 < t_len):
        raise ValueError("T0 out of range")
    if not np.isfinite(panel).all():
        raise ValueError("non-finite input")
    ratios = np.empty(n_units)
    for j in range(n_units):
        donors = np.delete(panel, j, axis=1)
        fit = synthetic_control(donors, panel[:, j], t0, n_iter=n_iter)
        ratios[j] = float(fit["rmspe_ratio"])
    return {
        "ratios": ratios,
        "treated_ratio": float(ratios[treated_idx]),
        "rank": int(np.sum(ratios >= ratios[treated_idx])),
        "p_value": float(np.mean(ratios >= ratios[treated_idx])),
    }
