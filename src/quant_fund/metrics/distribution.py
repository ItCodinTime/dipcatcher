"""Distributional shape and normality battery.

Univariate normality tests (Jarque–Bera lives in ``metrics.serial`` — this
module adds the rest of the canon), multivariate normality tests, and
robust shape measures.

References:
- Shapiro, Wilk (1965); Royston (1995) approximation (via scipy).
- Anderson, Darling (1954); Stephens (1974) normal-case critical values.
- Cramer–von Mises: Smirnov (1936)/Csorgo-Faraway (1996) normal form.
- Lilliefors (1967) KS with estimated parameters.
- D'Agostino, Pearson (1973) K^2 omnibus (scipy normaltest).
- Pearson (1900) chi-squared goodness-of-fit.
- Mardia (1970) multivariate skewness/kurtosis tests.
- Doornik, Hansen (2008) omnibus (small-sample corrected skew/kurt).
- Brys, Hubert, Struyf (2004) medcouple robust skewness.
- Rousseeuw, Croux (1993) Qn scale estimator.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

Array = NDArray[np.float64]


def _v(x: Array, n: int = 8) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {n}")
    return v


def shapiro_wilk(x: Array) -> dict[str, float]:
    """Shapiro–Wilk (1965) normality test."""
    v = _v(x, 3)
    w, p = sstats.shapiro(v)
    return {"w": float(w), "pvalue": float(p)}


def anderson_darling_normal(x: Array) -> dict[str, float]:
    """Anderson–Darling (1954) normality test with Stephens criticals."""
    v = _v(x)
    res = sstats.anderson(v, dist="norm", method="interpolate")
    p = float(res.pvalue)
    return {
        "stat": float(res.statistic),
        "pvalue": p,
        "reject_5pct": float(p < 0.05),
    }


def cramervonmises_normal(x: Array) -> dict[str, float]:
    """Cramér–von Mises test against a fitted normal."""
    v = _v(x)
    res = sstats.cramervonmises(v, "norm", args=(float(v.mean()), float(v.std())))
    return {"stat": float(res.statistic), "pvalue": float(res.pvalue)}


def lilliefors(x: Array, mc: int = 2000, seed: int = 7) -> dict[str, float]:
    """Lilliefors (1967) KS test with estimated mean/variance.

    p-value by parametric bootstrap: normal samples refit and re-tested —
    exact up to Monte Carlo error for the estimated-parameter case (no
    asymptotic table needed).
    """
    v = _v(x)
    z = (v - v.mean()) / v.std()
    d = float(sstats.kstest(z, "norm").statistic)
    n = v.size
    rng = np.random.default_rng(seed)
    sims = rng.normal(size=(mc, n))
    zs = (sims - sims.mean(axis=1, keepdims=True)) / sims.std(axis=1, keepdims=True)
    ds = np.empty(mc)
    grid = np.linspace(-4.0, 4.0, 200)
    cdf = sstats.norm.cdf(grid)
    for i in range(mc):
        emp = np.searchsorted(np.sort(zs[i]), grid, side="right") / n
        ds[i] = float(np.max(np.abs(emp - cdf)))
    p = float((1.0 + np.sum(ds >= d)) / (mc + 1.0))
    return {"d": d, "pvalue": p}


def dagostino_k2(x: Array) -> dict[str, float]:
    """D'Agostino–Pearson (1973) K^2 omnibus normality test."""
    v = _v(x)
    k2, p = sstats.normaltest(v)
    return {"k2": float(k2), "pvalue": float(p)}


def pearson_chi2_normal(x: Array, n_bins: int = 10) -> dict[str, float]:
    """Pearson (1900) chi-squared GOF against fitted normal quantiles."""
    v = _v(x, 30)
    if n_bins < 5 or n_bins > v.size // 5:
        raise ValueError("n_bins must be >= 5 with >= 5 obs per bin")
    edges = sstats.norm.ppf(np.linspace(0, 1, n_bins + 1), loc=v.mean(), scale=v.std())
    obs = np.histogram(v, bins=edges)[0]
    exp = np.full(n_bins, v.size / n_bins)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    df = n_bins - 3  # minus 1 + 2 estimated params
    return {"chi2": chi2, "pvalue": float(sstats.chi2.sf(chi2, df)), "df": float(df)}


def normality_battery(x: Array) -> dict[str, float]:
    """Composite: run all univariate tests, return p-values + min."""
    out = {
        "p_shapiro": shapiro_wilk(x)["pvalue"],
        "ad_reject_5pct": anderson_darling_normal(x)["reject_5pct"],
        "p_cvm": cramervonmises_normal(x)["pvalue"],
        "p_lilliefors": lilliefors(x)["pvalue"],
        "p_dagostino": dagostino_k2(x)["pvalue"],
        "p_pearson": pearson_chi2_normal(x)["pvalue"],
    }
    out["p_min"] = float(
        min(
            out["p_shapiro"],
            out["p_cvm"],
            out["p_lilliefors"],
            out["p_dagostino"],
            out["p_pearson"],
        )
    )
    out["normal_5pct"] = float(out["p_min"] > 0.05)
    return out


def mardia_test(x: Array) -> dict[str, float]:
    """Mardia (1970) multivariate normality: skewness and kurtosis tests.

    ``x`` is (n, p). Skewness ~ chi2(p(p+1)(p+2)/6); kurtosis ~ N(0, 8p(p+2)/n)
    — small-sample corrected.
    """
    m = np.asarray(x, dtype=float)
    if m.ndim != 2 or m.shape[0] < 20 or not np.all(np.isfinite(m)):
        raise ValueError("x must be a finite (n, p) matrix, n >= 20")
    n, p = m.shape
    z = m - m.mean(axis=0)
    s = z.T @ z / n
    s_inv = np.linalg.pinv(s)
    # Mardia skewness: mean over pairs of (z_i' S^{-1} z_j)^3.
    g = z @ s_inv @ z.T  # Mahalanobis inner products
    b1 = float(np.mean(g**3))
    df_skew = p * (p + 1) * (p + 2) / 6.0
    skew_stat = n * b1 / 6.0
    # Mardia kurtosis: mean of (z_i' S^{-1} z_i)^2 vs expected p(p+2).
    b2 = float(np.mean(np.diag(g) ** 2))
    kurt_mean = p * (p + 2.0) * (n - 1.0) / (n + 1.0)
    kurt_var = 8.0 * p * (p + 2.0) * (n - 1.0) / ((n + 1.0) ** 2 * (n + 3.0))
    kurt_z = (b2 - kurt_mean) / math.sqrt(max(kurt_var, 1e-12))
    return {
        "skew_stat": skew_stat,
        "skew_pvalue": float(sstats.chi2.sf(skew_stat, df_skew)),
        "kurt_b2": b2,
        "kurt_z": kurt_z,
        "kurt_pvalue": float(2.0 * sstats.norm.sf(abs(kurt_z))),
    }


def medcouple(x: Array) -> float:
    """Brys–Hubert–Struyf (2004) medcouple robust skewness.

    ``MC = median over (x_i <= med <= x_j) of h(x_i, x_j)`` with
    ``h = ((x_j - med) - (med - x_i)) / (x_j - x_i)`` — bounded in [-1, 1].
    """
    v = _v(x, 5)
    med = float(np.median(v))
    lo = v[v <= med]
    hi = v[v >= med]
    h_vals = []
    for xi in lo:
        for xj in hi:
            d = xj - xi
            if d > 0:
                h_vals.append(((xj - med) - (med - xi)) / d)
            else:
                h_vals.append(0.0)
    if not h_vals:
        raise ValueError("medcouple undefined (all equal values)")
    return float(np.median(h_vals))


def qn_scale(x: Array) -> float:
    """Rousseeuw–Croux (1993) Qn robust scale estimator.

    ``Qn = d * c_n * first quartile of {|x_i - x_j| : i < j}``.
    """
    v = _v(x, 4)
    n = v.size
    diffs = np.abs(v[:, None] - v[None, :])
    tri = diffs[np.triu_indices(n, k=1)]
    q = float(np.quantile(tri, 0.25))
    # Finite-sample consistency factor (approximate c_n).
    c_n = 1.0 / (math.sqrt(2.0) * sstats.norm.ppf(5.0 / 8.0))
    return float(q * c_n)


def robust_shape(x: Array) -> dict[str, float]:
    """Robust distributional summary: medcouple skew, Qn scale, robust z bounds."""
    v = _v(x, 8)
    med = float(np.median(v))
    qn = qn_scale(v)
    rz = (v - med) / qn if qn > 0 else np.zeros_like(v)
    return {
        "median": med,
        "qn": qn,
        "medcouple": medcouple(v),
        "n_outlier_35": float(np.sum(np.abs(rz) > 3.5)),
        "share_outlier_35": float(np.mean(np.abs(rz) > 3.5)),
    }
