"""Long-memory (fractional-integration) estimators.

References:
- Geweke & Porter-Hudak (1983): log-periodogram regression for d.
- Kunsch (1987) / Robinson (1995): local Whittle semiparametric d.
- Fox & Taqqu (1986): Whittle likelihood; used here for ARFIMA(0,d,0).
- Lo (1991): modified rescaled-range statistic.
- Robinson (1995): semiparametric Gaussian estimator consistency.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt
from scipy import stats

Array = NDArray[np.float64]


def _v(x: Array, n: int = 64) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {n}")
    if v.std() == 0:
        raise ValueError("degenerate (constant) series")
    return v


def _periodogram(v: Array) -> tuple[Array, Array]:
    """Fourier frequencies j = 1..m and periodogram I(lambda_j)."""
    n = v.size
    vc = v - v.mean()
    F = np.fft.rfft(vc)
    per = (np.abs(F) ** 2) / n
    lam = 2.0 * math.pi * np.arange(1, F.size) / n
    return lam, per[1:]


def gph_estimate(series: Array, bandwidth: float | None = None) -> dict[str, float]:
    """Geweke–Porter-Hudak (1983) log-periodogram regression.

    ``ln I(lambda_j) = c - d * ln(4 sin^2(lambda_j/2)) + eps`` on the
    lowest m frequencies (m ~ n^0.5 default). Standard error uses the
    GPH asymptotic variance pi^2/(6m).
    """
    v = _v(series)
    n = v.size
    lam, per = _periodogram(v)
    m = int(max(10, n**0.5)) if bandwidth is None else int(bandwidth)
    m = min(m, lam.size)
    lam_m = lam[:m]
    per_m = np.maximum(per[:m], 1e-300)
    x = np.log(4.0 * np.sin(lam_m / 2.0) ** 2)
    yv = np.log(per_m)
    X = np.column_stack([np.ones(m), x])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    d_hat = float(-beta[1])
    se = math.sqrt(math.pi**2 / (6.0 * m))
    return {
        "d": d_hat,
        "se": se,
        "t_iid": d_hat / se,
        "pvalue_iid": float(2.0 * (1.0 - stats.norm.cdf(abs(d_hat / se)))),
        "m": float(m),
    }


def local_whittle(series: Array, bandwidth: float | None = None) -> dict[str, float]:
    """Kunsch (1987)/Robinson (1995) local Whittle estimator.

    Minimizes ``R(d) = ln( (1/m) sum_j lambda_j^{2d} per_j ) - 2d/m sum ln lambda_j``
    over d in (-0.5, 1). Asymptotic se = 1/(2 sqrt(m)).
    """
    v = _v(series)
    n = v.size
    lam, per = _periodogram(v)
    m = int(max(10, n**0.6)) if bandwidth is None else int(bandwidth)
    m = min(m, lam.size)
    lam_m = lam[:m]
    per_m = np.maximum(per[:m], 1e-300)
    lnl = np.log(lam_m)

    def R(d: float) -> float:
        g = lam_m ** (2.0 * d) * per_m
        out = math.log(float(g.mean())) - 2.0 * d * float(lnl.mean())
        return out if np.isfinite(out) else 1e12

    res = opt.minimize_scalar(R, bounds=(-0.5, 1.0), method="bounded")
    d_hat = float(res.x)
    se = 1.0 / (2.0 * math.sqrt(m))
    return {
        "d": d_hat,
        "se": se,
        "t_iid": d_hat / se,
        "pvalue_iid": float(2.0 * (1.0 - stats.norm.cdf(abs(d_hat / se)))),
        "m": float(m),
    }


def whittle_arfima(series: Array) -> dict[str, float]:
    """Fox–Taqqu Whittle likelihood for ARFIMA(0, d, 0).

    Approximates the spectral density f(lambda) ~ |1 - e^{-i lambda}|^{-2d}
    and minimizes sum_j per_j / f_j over the full periodogram (valid for
    -0.5 < d < 0.5 stationary region).
    """
    v = _v(series)
    n = v.size
    lam, per = _periodogram(v)
    per = np.maximum(per, 1e-300)

    def R(d: float) -> float:
        # Whittle likelihood: minimize mean(per_j / f_j(d)); the ARFIMA
        # spectrum is already sigma^2-normalized since int log f = 0.
        f = np.abs(1.0 - np.exp(-1j * lam)) ** (-2.0 * d)
        out = float((per / f).mean())
        return out if np.isfinite(out) else 1e12

    res = opt.minimize_scalar(R, bounds=(-0.49, 0.49), method="bounded")
    d_hat = float(res.x)
    # Fox–Taqqu asymptotic variance: Var(d) = pi^2/(24n) -> se = pi/sqrt(24n).
    se = math.pi / math.sqrt(24.0 * n)
    return {
        "d": d_hat,
        "se": se,
        "t_iid": d_hat / se,
        "pvalue_iid": float(2.0 * (1.0 - stats.norm.cdf(abs(d_hat / se)))),
        "n": float(n),
    }


def lo_modified_rs(series: Array, q: int | None = None) -> dict[str, float]:
    """Lo (1991) modified rescaled-range statistic.

    ``Q_n = R/S`` where R is the range of cumulative demeaned sums and
    S is a HAC long-run std with lag window q ~ n^{1/4}. Under short
    memory Q_n/sqrt(n) ~ V in [Feller's range distribution]; Lo's table
    gives the ~5-95% interval [0.809, 1.862].
    """
    v = _v(series)
    n = v.size
    q = int(n**0.25) if q is None else int(q)
    if q < 1 or q >= n // 2:
        raise ValueError("lag window must be in [1, n/2)")
    vc = v - v.mean()
    R = float(np.ptp(np.cumsum(vc)))
    s2 = float(vc @ vc / n)
    for h in range(1, q + 1):
        w = 1.0 - h / (q + 1.0)
        s2 += 2.0 * w * float(np.dot(vc[h:], vc[:-h]) / n)
    s2 = max(s2, 1e-20)
    Qn = R / math.sqrt(s2)
    stat = Qn / math.sqrt(n)
    # Lo's tabulated ~[5%, 95%] acceptance band under short memory.
    return {
        "Q": Qn,
        "statistic": stat,
        "lo_5pct": 0.809,
        "lo_95pct": 1.862,
        "reject_short_memory": float(stat < 0.809 or stat > 1.862),
        "q": float(q),
        "n": float(n),
    }
