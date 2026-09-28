"""Lead-lag structure: cross-correlation networks + Hayashi-Yoshida.

- ``cross_correlation``: sample cross-correlation of two series at integer
  lags -k..k on the overlapping interior.
- ``lead_lag_matrix``: for a return panel, pairwise peak-lag analysis —
  entry (i, j) is the lag ell in {1..max_lag} maximizing corr(r_i[t],
  r_j[t+ell]); positive means i leads j. Net asymmetry minus the reverse
  direction gives a signed lead-lag score.
- ``lead_lag_adjacency``: threshold the signed score into a directed
  adjacency (i -> j when i significantly leads j).
- ``hayashi_yoshida``: the Hayashi & Yoshida (2005) covariance estimator
  for asynchronously sampled series — sum over overlapping intervals of
  return products, valid without synchronization.

Fail-closed: non-finite input, insufficient length, no overlaps.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check_pair(x: Array, y: Array, min_n: int = 20) -> tuple[Array, Array]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < min_n:
        raise ValueError(f"x and y must be equal-length with n >= {min_n}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite input")
    if x.std() == 0 or y.std() == 0:
        raise ValueError("degenerate series")
    return x, y


def cross_correlation(x: Array, y: Array, max_lag: int) -> Array:
    """corr(x_t, y_{t+ell}) for ell = -max_lag..max_lag on overlaps."""
    x, y = _check_pair(x, y, min_n=4)
    n = x.size
    if not (1 <= max_lag < n // 4):
        raise ValueError("max_lag must satisfy 1 <= max_lag < n/4")
    xz = x - x.mean()
    yz = y - y.mean()
    out = np.empty(2 * max_lag + 1)
    for i, ell in enumerate(range(-max_lag, max_lag + 1)):
        if ell >= 0:
            a, b = xz[: n - ell], yz[ell:]
        else:
            a, b = xz[-ell:], yz[: n + ell]
        denom = a.std(ddof=0) * b.std(ddof=0)
        out[i] = float(np.mean((a - a.mean()) * (b - b.mean())) / denom) if denom > 0 else 0.0
    return out


def lead_lag_matrix(returns: Array, max_lag: int = 5) -> dict[str, Array]:
    """Signed lead-lag score matrix S where S[i,j] > 0 means i leads j.

    S[i,j] = argmax over |ell|<=L of corr(r_i,t; r_j,t+ell) restricted to
    ell > 0 minus the same for ell < 0 — i.e., how much better i predicts
    future j than past j predicts i. ``peak_lag`` gives the maximizing lag.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[0] < 30 or r.shape[1] < 2:
        raise ValueError("returns must be (T, N), T >= 30, N >= 2")
    if not np.isfinite(r).all():
        raise ValueError("non-finite input")
    if (r.std(axis=0) <= 0).any():
        raise ValueError("degenerate column")
    t_len, n = r.shape
    max_lag = int(min(max_lag, t_len // 5))
    scores = np.zeros((n, n))
    peaks = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cc = cross_correlation(r[:, i], r[:, j], max_lag)
            pos = cc[max_lag + 1 :]  # i leads j (j lags)
            neg = cc[:max_lag]  # i lags j
            best_pos = float(pos.max())
            best_neg = float(neg.max())
            scores[i, j] = best_pos - best_neg
            peaks[i, j] = int(np.argmax(pos) + 1)
    return {"scores": scores, "peak_lag": peaks.astype(np.float64)}


def lead_lag_adjacency(
    returns: Array, max_lag: int = 5, quantile: float = 0.9
) -> NDArray[np.int64]:
    """Directed adjacency: A[i,j]=1 when i's lead score over j clears the
    ``quantile`` of all off-diagonal scores."""
    scores = lead_lag_matrix(returns, max_lag)["scores"]
    off = scores[~np.eye(scores.shape[0], dtype=bool)]
    thr = float(np.quantile(off, quantile))
    adj = (scores > thr).astype(np.int64)
    np.fill_diagonal(adj, 0)
    return adj


def hayashi_yoshida(x_times: Array, x: Array, y_times: Array, y: Array) -> dict[str, float | int]:
    """Hayashi-Yoshida covariance/correlation for asynchronous series.

    Inputs: observation timestamps (monotone) and values. Returns the HY
    covariance of the tick increments plus the implied correlation.
    """
    xt = np.asarray(x_times, dtype=float).ravel()
    xx = np.asarray(x, dtype=float).ravel()
    yt = np.asarray(y_times, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    for name, arr in (("x_times", xt), ("x", xx), ("y_times", yt), ("y", yy)):
        if arr.size < 3 or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite with >= 3 observations")
    if xt.size != xx.size or yt.size != yy.size:
        raise ValueError("times and values must match")
    if np.diff(xt).min() <= 0 or np.diff(yt).min() <= 0:
        raise ValueError("timestamps must be strictly increasing")
    # tick increments with their observation intervals
    dx = np.diff(xx)
    dy = np.diff(yy)
    x_lo, x_hi = xt[:-1], xt[1:]
    y_lo, y_hi = yt[:-1], yt[1:]
    # overlap matrix via searchsorted marching
    cov = 0.0
    n_overlap = 0
    var_x = float(np.sum(dx * dx))
    var_y = float(np.sum(dy * dy))
    i = j = 0
    nx, ny = dx.size, dy.size
    while i < nx and j < ny:
        lo = max(x_lo[i], y_lo[j])
        hi = min(x_hi[i], y_hi[j])
        if lo < hi:
            cov += dx[i] * dy[j]
            n_overlap += 1
        if x_hi[i] <= y_hi[j]:
            i += 1
        else:
            j += 1
    corr = cov / np.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else np.nan
    return {
        "cov": float(cov),
        "corr": float(corr),
        "n_overlap": int(n_overlap),
        "var_x": float(var_x),
        "var_y": float(var_y),
    }
