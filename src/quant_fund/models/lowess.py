"""LOWESS: locally weighted scatterplot smoothing (Cleveland 1979).

For each target point the fit is a weighted local linear regression over the
``frac`` nearest neighbours, weighted by the tricube kernel of scaled distance.
Robustness iterations then down-weight large residuals with Tukey's bisquare
weights, making the smoother resistant to outliers.

Reference: W. S. Cleveland (1979), "Robust locally weighted regression and
smoothing scatterplots", JASA.  Fail-closed on non-finite input or an invalid
span.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _tricube(u: Array) -> Array:
    w = np.clip(1.0 - np.abs(u) ** 3, 0.0, None) ** 3
    return w


def lowess(x: Array, y: Array, frac: float = 0.3, iters: int = 3) -> dict[str, Array]:
    """Robust LOWESS smoother; returns the fit evaluated at sorted ``x``."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size or xa.size < 5 or not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        raise ValueError("x and y must be finite, aligned, and length >= 5")
    if not 0.0 < frac <= 1.0:
        raise ValueError("frac must be in (0, 1]")
    order = np.argsort(xa)
    xs, ys = xa[order], ya[order]
    n = xs.size
    r = int(np.ceil(frac * n))
    r = min(max(r, 2), n)
    fitted = np.zeros(n)
    robust = np.ones(n)
    for _ in range(max(iters, 1)):
        for i in range(n):
            dist = np.abs(xs - xs[i])
            idx = np.argsort(dist)[:r]
            d = dist[idx]
            dmax = d.max() if d.max() > 0 else 1.0
            w = _tricube(d / dmax) * robust[idx]
            xw = xs[idx]
            yw = ys[idx]
            sw = w.sum()
            if sw <= 0.0:
                fitted[i] = yw.mean()
                continue
            # Weighted linear regression at xs[i].
            mx = np.sum(w * xw) / sw
            my = np.sum(w * yw) / sw
            sxx = np.sum(w * (xw - mx) ** 2)
            sxy = np.sum(w * (xw - mx) * (yw - my))
            beta = sxy / sxx if sxx > 0 else 0.0
            fitted[i] = my + beta * (xs[i] - mx)
        resid = ys - fitted
        s = np.median(np.abs(resid))
        if s <= 0.0:
            break
        robust = np.clip(1.0 - (resid / (6.0 * s)) ** 2, 0.0, None) ** 2
    return {"x": xs, "fitted": fitted}
