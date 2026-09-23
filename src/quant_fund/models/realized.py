"""Realized volatility measures and jump detection.

Intraday high-frequency estimators: plain realized variance, realized
kernels (BNHLS), two-scales subsampling (ZMA), pre-averaging, bipower/
tripower quarticity, Barndorff-Nielsen–Shephard jump test, Lee–Mykland
jump detection, and semivariance decomposition.

References:
- Andersen, Bollerslev, Diebold, Labys (2001) realized variance.
- Barndorff-Nielsen, Hansen, Lunde, Shephard (2008) realized kernels.
- Zhang, Mykland, Ait-Sahalia (2005) two-scales realized variance (TSRV).
- Jacod, Li, Mykland, Podolskij, Vetter (2009) pre-averaging.
- Barndorff-Nielsen, Shephard (2004/2006) bipower variation jump test.
- Lee, Mykland (2008) jump test with bipower-robust normalization.
- Barndorff-Nielsen, Kinnebrock, Shephard (2010) semivariance.
- Podolskij, Vetter (2009) bipower/tripower quarticity estimators.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

Array = NDArray[np.float64]


def _as_returns(r: Array, min_len: int = 10) -> Array:
    v = np.asarray(r, dtype=float).reshape(-1)
    if v.size < min_len or not np.all(np.isfinite(v)):
        raise ValueError(f"returns must be finite with length >= {min_len}")
    return v


def realized_variance(r: Array) -> float:
    """Sum of squared intraperiod returns (ABDL 2001)."""
    v = _as_returns(r)
    return float(v @ v)


def bipower_variation(r: Array) -> float:
    """Barndorff-Nielsen–Shephard bipower variation.

    ``BV = mu1^{-2} * (n/(n-1)) * sum |r_t| * |r_{t-1}|`` — consistent for
    integrated variance under jumps.
    """
    v = _as_returns(r)
    mu1 = math.sqrt(2.0 / math.pi)
    n = v.size
    return float(np.sum(np.abs(v[1:]) * np.abs(v[:-1])) * (n / (n - 1)) / mu1**2)


def tripower_quarticity(r: Array) -> float:
    """Podolskij–Vetter (2009) tripower quarticity — jump-robust IQ estimator.

    ``TPQ = n * mu43^{-3} * sum |r_t|^{4/3}|r_{t-1}|^{4/3}|r_{t-2}|^{4/3}``
    """
    v = _as_returns(r, min_len=12)
    mu43 = 2.0 ** (2.0 / 3.0) * math.gamma(7.0 / 6.0) / math.gamma(0.5)
    n = v.size
    a = np.abs(v)
    tp = np.sum(a[2:] ** (4.0 / 3.0) * a[1:-1] ** (4.0 / 3.0) * a[:-2] ** (4.0 / 3.0))
    return float(n * tp * (n / (n - 2)) / mu43**3)


def bns_jump_test(r: Array, alpha: float = 0.999) -> dict[str, float]:
    """Barndorff-Nielsen–Shephard relative jump test.

    ``RJ = (RV - BV) / RV``; z-stat uses tripower-quarticity-scaled
    variance of the ratio (Huang–Tauchen 2005 relative-jump form).
    Returns the test statistic, p-value (one-sided: jump presence), and
    the jump/continuous variance shares.
    """
    v = _as_returns(r, min_len=20)
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must be in (0.5, 1)")
    rv = realized_variance(v)
    bv = bipower_variation(v)
    if rv <= 0.0 or bv <= 0.0:
        raise ValueError("degenerate variance estimators")
    tpq = tripower_quarticity(v)
    n = v.size
    # Huang–Tauchen (2005) z: sqrt(n) * RJ / sqrt(c * max(1, TPQ/BV^2)),
    # c = (pi/2)^2 + pi - 5 ~ 2.467.
    c = (math.pi / 2.0) ** 2 + math.pi - 5.0
    ratio = max(1.0, tpq / (bv * bv))
    rj = (rv - bv) / rv
    z = math.sqrt(n) * rj / math.sqrt(c * ratio)
    return {
        "z": z,
        "pvalue": float(sstats.norm.sf(z)),
        "rj": rj,
        "jump_share": max(rj, 0.0),
        "rv": rv,
        "bv": bv,
    }


def lee_mykland_jumps(r: Array, alpha: float = 0.999) -> dict[str, Array]:
    """Lee–Mykland (2008) instantaneous jump test.

    Test statistic ``L_t = r_t / sigma_t`` with ``sigma_t`` the bipower
    local volatility over window K ~ sqrt(n). Rejection threshold uses the
    Gumbel law ``(S_n - C_n)/s_n`` where
    ``C_n = sqrt(2 ln n) - (ln pi + ln ln n)/(2 sqrt(2 ln n))``, ``s_n = 1/sqrt(2 ln n)``.
    Returns jump indicators and the L statistic series.
    """
    v = _as_returns(r, min_len=20)
    n = v.size
    k = max(2, int(math.floor(math.sqrt(n))))
    sigma2 = np.full(n, np.nan)
    for t in range(k, n):
        seg = v[t - k : t]
        bp = np.sum(np.abs(seg[1:]) * np.abs(seg[:-1]))
        sigma2[t] = bp * math.pi / (2.0 * (k - 1))  # per-step bipower IV
    stat = np.abs(v) / np.sqrt(np.maximum(sigma2, 1e-16))
    stat = np.where(np.isfinite(stat), stat, 0.0)
    c_n = math.sqrt(2.0 * math.log(n)) - (math.log(math.pi) + math.log(math.log(n))) / (
        2.0 * math.sqrt(2.0 * math.log(n))
    )
    s_n = 1.0 / math.sqrt(2.0 * math.log(n))
    beta_star = -math.log(-math.log(alpha))
    thresh = beta_star * s_n + c_n
    jumps = stat > thresh
    return {
        "is_jump": jumps,
        "stat": stat,
        "threshold": np.array([thresh]),
        "jump_idx": np.flatnonzero(jumps).astype(float),
    }


def tsrv(r: Array, n_grids: int | None = None) -> float:
    """Zhang–Mykland–Ait-Sahalia (2005) two-scales realized variance.

    Splits the price path into K non-overlapping sparse grids, averages
    their realized variances, then subtracts the noise-bias correction
    ``(nbar/n) * RV_all``. Default K ~ n^{2/3} (ZMA optimal rate).
    """
    v = _as_returns(r, min_len=20)
    n = v.size
    if n_grids is None:
        n_grids = max(2, int(round(n ** (2.0 / 3.0))))
    if n_grids < 1 or n_grids >= n // 2:
        raise ValueError("n_grids out of range")
    cum = np.concatenate([[0.0], np.cumsum(v)])
    rvs = []
    for g in range(n_grids):
        idx = np.arange(g, n + 1, n_grids)
        if idx[-1] != n:
            idx = np.append(idx, n)
        if idx.size > 2:
            diffs = np.diff(cum[idx])
            rvs.append(float(diffs @ diffs))
    if not rvs:
        raise ValueError("no sub-grids formed")
    n_bar = n / n_grids
    return float(np.mean(rvs) - (n_bar / n) * realized_variance(v))


def realized_kernel(r: Array, bandwidth: int | None = None) -> float:
    """Barndorff-Nielsen–Hansen–Lunde–Shephard (2008) flat-top realized kernel.

    ``K = RV + 2 * sum_h w(h/H) * gamma_h`` with Parzen weights and
    ``gamma_h = sum_t r_t r_{t-h}``.
    """
    v = _as_returns(r, min_len=20)
    n = v.size
    if bandwidth is None:
        bandwidth = max(1, int(round(3.51 * n**0.2)))
    if bandwidth < 1 or bandwidth >= n:
        raise ValueError("bandwidth out of range")
    rv = realized_variance(v)
    adj = 0.0
    for h in range(1, bandwidth + 1):
        x = h / (bandwidth + 1.0)
        w = 1.0 - 6.0 * x * x + 6.0 * x**3 if x <= 0.5 else 2.0 * (1.0 - x) ** 3
        gamma = float(v[h:] @ v[:-h])
        adj += w * gamma
    return rv + 2.0 * adj


def preaveraged_rv(r: Array, theta: float = 0.4) -> float:
    """Jacod et al. (2009) pre-averaging realized variance (modulated RV).

    Weight function ``g(x) = min(x, 1-x)`` with psi1 = int g'^2 = 1 and
    psi2 = int g^2 = 1/12; window ``k_n = theta*sqrt(n)``.
    """
    v = _as_returns(r, min_len=30)
    n = v.size
    if theta <= 0 or not np.isfinite(theta):
        raise ValueError("theta must be positive")
    k_n = max(3, int(round(theta * math.sqrt(n))))
    m = n - k_n + 1
    if m <= 1:
        raise ValueError("series too short for pre-averaging window")
    psi1, psi2 = 1.0, 1.0 / 12.0
    jg = np.arange(1, k_n) / k_n
    w = np.minimum(jg, 1.0 - jg)
    rbar = np.array([float(w @ v[t + 1 : t + k_n]) for t in range(m)])
    mrv = (n / m) * float(rbar @ rbar) / (psi2 * k_n)
    return float(mrv - psi1 * realized_variance(v) / (2.0 * theta**2 * psi2 * n))


def semivariance(r: Array) -> dict[str, float]:
    """Barndorff-Nielsen–Kinnebrock–Shephard (2010) up/down semivariance."""
    v = _as_returns(r)
    pos = v[v > 0]
    neg = v[v < 0]
    rs_p = float(pos @ pos)
    rs_n = float(neg @ neg)
    total = rs_p + rs_n
    return {
        "rs_pos": rs_p,
        "rs_neg": rs_n,
        "signed_jump": rs_p - rs_n,
        "rs_ratio": (rs_p - rs_n) / total if total > 0 else np.nan,
    }


def realized_quarticity(r: Array) -> float:
    """Plain realized quarticity ``(n/3) * sum r_t^4`` — consistent IQ under
    no jumps (used for RV confidence bands, BNS 2002).
    """
    v = _as_returns(r)
    n = v.size
    return float(n * np.sum(v**4) / 3.0)


def rv_confidence_band(r: Array, alpha: float = 0.95, robust: bool = True) -> dict[str, float]:
    """BNS (2002) asymptotic log-RV confidence band.

    ``log RV ± z * sqrt(2/3 * RQ/RV^2 / n)``; ``robust`` uses tripower
    quarticity (valid under jumps), else realized quarticity.
    """
    v = _as_returns(r, min_len=20)
    rv = realized_variance(v)
    if rv <= 0.0:
        raise ValueError("zero realized variance")
    rq = tripower_quarticity(v) if robust else realized_quarticity(v)
    n = v.size
    var_log = 2.0 / 3.0 * rq / (rv * rv * n)
    z = sstats.norm.ppf(0.5 + alpha / 2.0)
    half = z * math.sqrt(max(var_log, 0.0))
    lo = math.exp(math.log(rv) - half)
    hi = math.exp(math.log(rv) + half)
    return {"rv": rv, "lower": lo, "upper": hi, "z": z}
