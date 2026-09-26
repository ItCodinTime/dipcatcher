"""Heteroskedasticity- and autocorrelation-consistent (HAC) inference.

Long-run variance estimators and HAC t-statistics for loss differentials and
mean tests — the machinery Diebold–Mariano-style comparisons need once
overlapping or serially correlated losses are involved.

References:
- Newey, West (1987). A simple, positive semi-definite, heteroskedasticity and
  autocorrelation consistent covariance matrix. *Econometrica* 55 — Bartlett.
- Andrews (1991). Heteroskedasticity and autocorrelation consistent covariance
  matrix estimation. *Econometrica* 59 — kernel bandwidth selection.
- Andrews, Monahan (1992). An improved HAC covariance matrix. *Econometrica*
  60 — AR(1) prewhitening with QS kernel.
- Gallant (1987) Parzen kernel; Priestley — quadratic spectral kernel.
- Diebold, Mariano (1995) — HAC is the standard fix for the loss differential.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

Array = NDArray[np.float64]

_KERNELS = ("bartlett", "quadratic_spectral", "parzen")


def _as_vector(x: Array, name: str = "x", *, min_obs: int = 8) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < min_obs:
        raise ValueError(f"{name} must contain at least {min_obs} finite observations")
    return v


def _autocovs(x: Array, max_lag: int) -> Array:
    e = x - x.mean()
    n = e.size
    gamma = np.empty(max_lag + 1)
    gamma[0] = float(np.dot(e, e) / n)
    for j in range(1, max_lag + 1):
        gamma[j] = float(np.dot(e[j:], e[:-j]) / n)
    return gamma


def _kernel_weight(kernel: str, z: float) -> float:
    if kernel == "bartlett":
        return max(0.0, 1.0 - abs(z))
    if kernel == "parzen":
        az = abs(z)
        if az <= 0.5:
            return 1.0 - 6.0 * az**2 + 6.0 * az**3
        if az <= 1.0:
            return 2.0 * (1.0 - az) ** 3
        return 0.0
    # quadratic spectral (Andrews QS): k(z) = 25/(12 pi^2 z^2) * (sin(m)/m - cos(m))
    if z == 0.0:
        return 1.0
    m = 6.0 * np.pi * z / 5.0
    return 25.0 / (12.0 * np.pi**2 * z**2) * (math.sin(m) / m - math.cos(m))


def kernel_lrv(x: Array, kernel: str = "bartlett", lag: int | None = None) -> float:
    """Kernel long-run variance ``gamma_0 + 2 sum_j w_j gamma_j``.

    ``lag`` is the truncation/bandwidth parameter; default ``floor(4 (T/100)^(2/9))``
    (Newey–West rule of thumb).  QS uses ``z = j / (lag + 1)``; Bartlett/Parzen
    use ``z = j / (lag + 1)`` as the scaled lag too.
    """
    v = _as_vector(x)
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}")
    if lag is None:
        lag = max(1, int(math.floor(4.0 * (v.size / 100.0) ** (2.0 / 9.0))))
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise ValueError("lag must be a positive integer")
    lag = min(lag, v.size - 2)
    gamma = _autocovs(v, lag)
    lrv = gamma[0]
    scale = lag + 1.0
    for j in range(1, lag + 1):
        lrv += 2.0 * _kernel_weight(kernel, j / scale) * gamma[j]
    return float(max(lrv, 0.0))


def newey_west_lrv(x: Array, lag: int | None = None) -> float:
    """Newey–West (1987) Bartlett-kernel LRV."""
    return kernel_lrv(x, "bartlett", lag)


def andrews_bandwidth(x: Array, kernel: str = "quadratic_spectral") -> float:
    """Andrews (1991) automatic bandwidth via approximating AR(1) model.

    Fits rho on demeaned x, then applies the AR(1)-based optimal bandwidth
    formula for the requested kernel (Bartlett: 1.1447*(alpha*T)^(1/3); QS:
    1.3221*(alpha*T)^(1/5); Parzen: 2.6614*(alpha*T)^(1/5)).
    """
    v = _as_vector(x)
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}")
    e = v - v.mean()
    denom = float(np.dot(e[:-1], e[:-1]))
    rho = float(np.dot(e[1:], e[:-1]) / denom) if denom > 0.0 else 0.0
    rho = float(np.clip(rho, -0.99, 0.99))
    t = float(v.size)
    # AR(1) alpha parameters (Andrews 1991, eq. 6.2/6.4 reduced form).
    alpha1 = 4.0 * rho**2 / (1.0 - rho) ** 4
    alpha2 = 4.0 * rho**2 * (1.0 + rho) ** 2 / (1.0 - rho) ** 8
    if kernel == "bartlett":
        bw = 1.1447 * (alpha1 * t) ** (1.0 / 3.0)
    elif kernel == "parzen":
        bw = 2.6614 * (alpha2 * t) ** (1.0 / 5.0)
    else:
        bw = 1.3221 * (alpha2 * t) ** (1.0 / 5.0)
    return float(max(1.0, min(bw, t - 2.0)))


def andrews_monahan_lrv(x: Array, lag: int | None = None) -> float:
    """Andrews–Monahan (1992) AR(1)-prewhitened QS long-run variance.

    Filters out an AR(1) component, estimates the QS LRV of residuals, then
    recolors by ``1/(1-rho)^2`` — the standard fix for size distortion in HAC.
    """
    v = _as_vector(x)
    e = v - v.mean()
    denom = float(np.dot(e[:-1], e[:-1]))
    rho = float(np.clip(np.dot(e[1:], e[:-1]) / denom, -0.99, 0.99)) if denom > 0.0 else 0.0
    resid = e[1:] - rho * e[:-1]
    bw = int(round(andrews_bandwidth(resid, "quadratic_spectral"))) if lag is None else lag
    lrv_resid = kernel_lrv(resid, "quadratic_spectral", max(1, bw))
    return float(lrv_resid / (1.0 - rho) ** 2)


def hac_mean_test(x: Array, mean0: float = 0.0, *, kernel: str = "bartlett") -> dict[str, float]:
    """HAC t-test of ``E[x] = mean0`` using the kernel LRV."""
    if not np.isfinite(mean0):
        raise ValueError("mean0 must be finite")
    v = _as_vector(x)
    lrv = kernel_lrv(v, kernel)
    se = math.sqrt(lrv / v.size)
    if se <= 0.0:
        return {"t": float("nan"), "pvalue": float("nan"), "lrv": lrv, "se": se}
    t = (float(v.mean()) - mean0) / se
    return {"t": t, "pvalue": float(2.0 * sstats.norm.sf(abs(t))), "lrv": lrv, "se": se}


def dm_hac_tstat(d: Array, *, prewhiten: bool = True) -> dict[str, float]:
    """Diebold–Mariano-style t-statistic on a loss differential ``d``.

    Uses the Andrews–Monahan prewhitened LRV by default (better size than plain
    Newey–West under strong autocorrelation); ``prewhiten=False`` gives the
    classic Newey–West stat.
    """
    v = _as_vector(d, "d")
    lrv = andrews_monahan_lrv(v) if prewhiten else newey_west_lrv(v)
    se = math.sqrt(lrv / v.size)
    if se <= 0.0:
        return {"t": float("nan"), "pvalue": float("nan"), "lrv": lrv, "se": se}
    t = float(v.mean()) / se
    return {"t": t, "pvalue": float(2.0 * sstats.norm.sf(abs(t))), "lrv": lrv, "se": se}


def hac_mean_covariance(x: Array, kernel: str = "bartlett", lag: int | None = None) -> Array:
    """HAC covariance of the column means of a (T, K) panel.

    ``cov = (1/T) * sum_h w_h (Gamma_h + Gamma_h')`` — the multivariate
    Newey–West form used for joint mean tests on loss-differential panels.
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 2 or x_arr.shape[0] < 8 or x_arr.shape[1] < 1:
        raise ValueError("x must be a (T, K) panel with T >= 8")
    x_arr = x_arr[np.isfinite(x_arr).all(axis=1)]
    if x_arr.shape[0] < 8:
        raise ValueError("too few complete rows")
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}")
    t = x_arr.shape[0]
    if lag is None:
        lag = max(1, int(math.floor(4.0 * (t / 100.0) ** (2.0 / 9.0))))
    lag = min(lag, t - 2)
    e = x_arr - x_arr.mean(axis=0)
    cov = e.T @ e / t
    for h in range(1, lag + 1):
        gamma_h = e[h:].T @ e[:-h] / t
        w = _kernel_weight(kernel, h / (lag + 1.0))
        cov += w * (gamma_h + gamma_h.T)
    return np.asarray(cov, dtype=float)
