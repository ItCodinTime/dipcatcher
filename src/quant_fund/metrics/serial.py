"""Serial-dependence, randomness, and unit-root diagnostics.

Classical battery for testing the iid / martingale assumptions that forecast
and risk benches rely on.  All estimators fail closed on degenerate input and
return honest ``nan`` statistics (never exceptions) when a test is well-posed
but numerically undefined for the observed sample.

References:
- Lo, MacKinlay (1988). Stock Market Prices Do Not Follow Random Walks. *RFS*.
- Chow, Denning (1993). A simple multiple variance ratio test. *J. Econometrics*.
- Wright (2000). Alternative variance-ratio tests using ranks and signs.
  *JBES* 18(1).
- Wald, Wolfowitz (1940). On a test whether two samples are from the same
  population. *Ann. Math. Statist.* (runs test).
- Ljung, Box (1978). On a measure of lack of fit in time series models.
  *Biometrika* 65.
- Box, Pierce (1970). Distribution of residual autocorrelations. *JASA* 65.
- Engle (1982). ARCH with estimates of the variance of UK inflation.
  *Econometrica* 50 — LM test for ARCH effects.
- Jarque, Bera (1980). Efficient tests for normality, homoscedasticity and
  serial independence. *Economics Letters* 6.
- Dickey, Fuller (1979); Phillips, Perron (1988); Kwiatkowski et al. (1992);
  Elliott, Rothenberg, Stock (1996) DF-GLS; Zivot, Andrews (1992) — unit-root
  family routed through :mod:`arch.unitroot` (already a repo dependency).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

Array = NDArray[np.float64]

_MIN_OBS = 4


def _as_vector(x: Array, name: str = "x", *, min_obs: int = _MIN_OBS) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < min_obs:
        raise ValueError(f"{name} must contain at least {min_obs} finite observations")
    return v


def autocorrelation(x: Array, max_lag: int = 20, *, demean: bool = True) -> Array:
    """Sample ACF ``rho_1..rho_max_lag`` (lag-0 excluded)."""
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 1:
        raise ValueError("max_lag must be a positive integer")
    v = _as_vector(x)
    z = v - v.mean() if demean else v.copy()
    denom = float(np.dot(z, z))
    if denom <= 0.0:
        raise ValueError("x must have positive variance")
    max_lag = min(max_lag, v.size - 1)
    out = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        out[k - 1] = float(np.dot(z[k:], z[:-k]) / denom)
    return out


def box_pierce(x: Array, lag: int = 10) -> dict[str, float]:
    """Box–Pierce ``Q = n * sum(rho_k^2)``, chi2(lag) asymptotics."""
    v = _as_vector(x, min_obs=lag + 2)
    rho = autocorrelation(v, lag)
    q = v.size * float(np.sum(rho**2))
    return {"stat": q, "pvalue": float(sstats.chi2.sf(q, lag)), "df": float(lag)}


def ljung_box(x: Array, lag: int = 10) -> dict[str, float]:
    """Ljung–Box ``Q = n(n+2) sum rho_k^2/(n-k)``, chi2(lag) asymptotics."""
    v = _as_vector(x, min_obs=lag + 2)
    rho = autocorrelation(v, lag)
    n = float(v.size)
    k = np.arange(1, lag + 1, dtype=float)
    q = n * (n + 2.0) * float(np.sum(rho**2 / (n - k)))
    return {"stat": q, "pvalue": float(sstats.chi2.sf(q, lag)), "df": float(lag)}


def engle_arch_lm(x: Array, lags: int = 5) -> dict[str, float]:
    """Engle (1982) ARCH LM test: ``T * R^2`` of squared residuals on lagged
    squares, chi2(lags)."""
    if isinstance(lags, bool) or not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer")
    v = _as_vector(x, min_obs=lags + 4)
    e = v - v.mean()
    y = e**2
    t = y.size - lags
    dep = y[lags:]
    reg = np.column_stack([np.ones(t)] + [y[lags - j - 1 : t + lags - j - 1] for j in range(lags)])
    beta, *_ = np.linalg.lstsq(reg, dep, rcond=None)
    resid = dep - reg @ beta
    ss_tot = float(np.sum((dep - dep.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else 1.0 - float(np.sum(resid**2)) / ss_tot
    lm = t * r2
    return {"stat": float(lm), "pvalue": float(sstats.chi2.sf(lm, lags)), "df": float(lags)}


def jarque_bera(x: Array) -> dict[str, float]:
    """Jarque–Bera normality test ``S^2/6 + (K-3)^2/24``, chi2(2)."""
    v = _as_vector(x)
    s = float(sstats.skew(v))
    k = float(sstats.kurtosis(v))  # excess kurtosis
    n = float(v.size)
    jb = n / 6.0 * (s**2 + 0.25 * k**2)
    return {"stat": jb, "pvalue": float(sstats.chi2.sf(jb, 2)), "skew": s, "excess_kurtosis": k}


def variance_ratio_test(x: Array, q: int = 2, *, log_prices: bool = False) -> dict[str, float]:
    """Lo–MacKinlay (1988) variance-ratio test on a return (or price) series.

    ``vr = Var(q-period return) / (q * Var(1-period return))``.  Returns the
    homoskedastic ``z1`` and heteroskedasticity-robust ``z2`` statistics with
    two-sided normal p-values.  ``log_prices=True`` differences the input first
    (classical convention: input is the log-price level).
    """
    if isinstance(q, bool) or not isinstance(q, int) or q < 2:
        raise ValueError("q must be an integer >= 2")
    v = _as_vector(x, min_obs=2 * q + 2)
    if log_prices:
        v = np.diff(v)
    n = v.size
    if n < 2 * q + 1:
        raise ValueError("series too short for requested q")
    mu = float(v.mean())
    e = v - mu
    s2_a = float(np.dot(e, e) / (n - 1))  # unbiased 1-period variance
    if s2_a <= 0.0:
        raise ValueError("x must have positive variance")
    m = q * (n - q + 1) * (1.0 - q / n)
    sums = np.convolve(v, np.ones(q), mode="valid") - q * mu
    s2_c = float(np.dot(sums, sums) / m)
    vr = s2_c / s2_a
    # Homoskedastic asymptotic variance of (VR - 1).
    phi = 2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * n)
    z1 = (vr - 1.0) / math.sqrt(phi)
    # Heteroskedastic-consistent: theta = sum_j [2(q-j)/q]^2 * delta_j with
    # delta_j the weighted squared autocovariance ratio (Lo–MacKinlay eq. 22).
    denom = float(np.sum(e**2) ** 2)
    theta = 0.0
    for j in range(1, q):
        # delta_j already scales as 1/n under iid — no extra factor of n.
        delta = float(np.sum(e[j:] ** 2 * e[:-j] ** 2)) / denom
        theta += (2.0 * (q - j) / q) ** 2 * delta
    z2 = (vr - 1.0) / math.sqrt(theta) if theta > 0.0 else float("nan")
    return {
        "vr": vr,
        "z1": z1,
        "pvalue_z1": float(2.0 * sstats.norm.sf(abs(z1))),
        "z2": z2,
        "pvalue_z2": float(2.0 * sstats.norm.sf(abs(z2))) if math.isfinite(z2) else float("nan"),
        "q": float(q),
        "n": float(n),
    }


def chow_denning_test(x: Array, periods: tuple[int, ...] = (2, 4, 8, 16)) -> dict[str, float]:
    """Chow–Denning (1993) joint variance-ratio test.

    Studentized maximum modulus of the robust ``z2`` statistics across
    ``periods``; p-value from the SMM bound ``1 - (2 Phi(z) - 1)^K``.
    """
    if not periods:
        raise ValueError("periods must be non-empty")
    zs: list[float] = []
    for q in periods:
        out = variance_ratio_test(x, q=q)
        z = out["z2"]
        if math.isfinite(z):
            zs.append(abs(z))
    if not zs:
        raise ValueError("no finite variance-ratio statistics")
    cd = max(zs)
    p = 1.0 - (2.0 * sstats.norm.cdf(cd) - 1.0) ** len(zs)
    return {"stat": cd, "pvalue": float(p), "k": float(len(zs))}


def wright_variance_ratio(
    x: Array, q: int = 2, *, n_boot: int = 0, seed: int = 0
) -> dict[str, float]:
    """Wright (2000) rank/sign-based variance-ratio tests.

    ``r1`` uses van der Waerden scores ``Phi^-1(rank/(T+1))``; ``r2`` uses
    ``score * sign(x - median)``.  Both statistics are VR-shaped ratios; the
    null distributions are non-standard, so ``n_boot > 0`` resamples iid signs
    of the observed scores to produce bootstrap p-values.
    """
    if isinstance(q, bool) or not isinstance(q, int) or q < 2:
        raise ValueError("q must be an integer >= 2")
    v = _as_vector(x, min_obs=2 * q + 2)
    t = v.size
    ranks = sstats.rankdata(v) / (t + 1.0)
    r1 = sstats.norm.ppf(ranks)
    r2 = r1 * np.sign(v - np.median(v))
    r2 = np.where(r2 == 0.0, 1e-12, r2)  # exact-median ties keep a sign

    def _vr(scores: Array) -> float:
        num = float(np.sum(np.convolve(scores, np.ones(q), mode="valid") ** 2)) / (t * q)
        den = float(np.sum(scores**2)) / t
        return num / den if den > 0.0 else float("nan")

    out = {"r1": _vr(r1), "r2": _vr(r2), "q": float(q), "n": float(t)}
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        b1 = np.empty(n_boot)
        b2 = np.empty(n_boot)
        for b in range(n_boot):
            signs = rng.choice([-1.0, 1.0], size=t)
            b1[b] = _vr(r1 * signs)  # iid-scramble ranks
            b2[b] = _vr(np.abs(r1) * signs)
        out["pvalue_r1"] = float(np.mean(b1 <= out["r1"]))
        out["pvalue_r2"] = float(np.mean(b2 <= out["r2"]))
    return out


def runs_test(x: Array, threshold: float | None = None) -> dict[str, float]:
    """Wald–Wolfowitz runs test on signs of ``x - threshold`` (default median).

    Under the null of random ordering the run count is approximately normal;
    returns the count, expected value and two-sided normal p-value.
    """
    v = _as_vector(x)
    thr = float(np.median(v)) if threshold is None else float(threshold)
    if not math.isfinite(thr):
        raise ValueError("threshold must be finite")
    signs = np.sign(v - thr)
    signs = signs[signs != 0.0]
    n1 = float(np.sum(signs > 0.0))
    n2 = float(np.sum(signs < 0.0))
    if n1 < 1.0 or n2 < 1.0:
        raise ValueError("runs test requires observations on both sides of threshold")
    runs = float(1.0 + np.sum(signs[1:] != signs[:-1]))
    e_runs = 1.0 + 2.0 * n1 * n2 / (n1 + n2)
    var_runs = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1.0))
    z = (runs - e_runs) / math.sqrt(var_runs) if var_runs > 0.0 else float("nan")
    return {
        "runs": runs,
        "expected_runs": e_runs,
        "z": z,
        "pvalue": float(2.0 * sstats.norm.sf(abs(z))) if math.isfinite(z) else float("nan"),
    }


def sign_test(x: Array) -> dict[str, float]:
    """Exact binomial sign test on the share of positive returns."""
    v = _as_vector(x)
    pos = int(np.sum(v > 0.0))
    neg = int(np.sum(v < 0.0))
    if pos + neg == 0:
        raise ValueError("x has no nonzero observations")
    p = float(sstats.binomtest(pos, pos + neg, 0.5).pvalue)
    return {
        "positive": float(pos),
        "negative": float(neg),
        "share_positive": pos / (pos + neg),
        "pvalue": p,
    }


def bartels_rank_test(x: Array) -> dict[str, float]:
    """Bartels (1982) rank test of randomness — distribution-free RNG check."""
    v = _as_vector(x)
    n = v.size
    ranks = sstats.rankdata(v)
    r_bar = 0.5 * (n + 1.0)
    num = float(np.sum((ranks[:-1] - ranks[1:]) ** 2))
    den = float(np.sum((ranks - r_bar) ** 2))
    rvn = num / den if den > 0.0 else float("nan")
    mean = 2.0
    var = 4.0 * (n - 2.0) / (n - 1.0)
    z = (rvn - mean) / math.sqrt(var) if n > 2 else float("nan")
    return {
        "stat": rvn,
        "z": z,
        "pvalue": float(2.0 * sstats.norm.sf(abs(z))) if math.isfinite(z) else float("nan"),
    }


def unit_root_battery(
    x: Array, *, trend: str = "c", lags: int | None = None
) -> dict[str, dict[str, float]]:
    """Unit-root / stationarity battery routed through :mod:`arch.unitroot`.

    Returns per-test ``{stat, pvalue}`` for ADF (Dickey–Fuller 1979), DF-GLS
    (Elliott–Rothenberg–Stock 1996), Phillips–Perron (1988), KPSS
    (Kwiatkowski–Phillips–Schmidt–Shin 1992 — note the *reversed* null of
    stationarity), and Zivot–Andrews (1992, one endogenous break).  Individual
    test failures degrade to ``nan`` entries rather than raising.
    """
    from arch.unitroot import ADF, DFGLS, KPSS, PhillipsPerron, ZivotAndrews

    v = _as_vector(x, min_obs=16)
    out: dict[str, dict[str, float]] = {}
    for name, cls, kwargs in (
        ("adf", ADF, {"trend": trend}),
        ("dfgls", DFGLS, {"trend": "c" if trend == "n" else trend}),
        ("phillips_perron", PhillipsPerron, {"trend": trend}),
        ("kpss", KPSS, {"trend": trend}),
        ("zivot_andrews", ZivotAndrews, {"trend": "c" if trend == "n" else trend}),
    ):
        kw: dict[str, Any] = dict(kwargs)
        if lags is not None:
            kw["lags"] = lags
        try:
            res = cls(v, **kw)
            out[name] = {"stat": float(res.stat), "pvalue": float(res.pvalue)}
        except Exception:  # per-test degradation: report nan, never raise
            out[name] = {"stat": float("nan"), "pvalue": float("nan")}
    return out
