"""PhD-level inference: HAC, Diebold–Mariano, bootstrap CIs, FDR.

All p-values are two-sided unless noted. Loss convention for DM is
user-supplied (smaller is better). See docs/MATH_SPEC.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def newey_west_variance(x: Array, lags: int | None = None) -> float:
    """Bartlett-kernel HAC variance of the sample mean of x.

    Omega = gamma_0 + 2 sum_{j=1}^L (1 - j/(L+1)) gamma_j
    Var(mean) = Omega / n
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return float("nan")
    xc = x - float(np.mean(x))
    if lags is None:
        lags = max(1, int(np.floor(1.5 * n ** (1.0 / 3.0))))
    lags = min(lags, n - 2)
    gamma0 = float(np.dot(xc, xc) / n)
    omega = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(np.dot(xc[lag:], xc[:-lag]) / n)
        omega += 2.0 * weight * gamma
    omega = max(omega, 1e-18)
    return omega / n


def newey_west_se(x: Array, lags: int | None = None) -> float:
    var = newey_west_variance(x, lags)
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(np.sqrt(var))


def mean_tstat(x: Array, lags: int | None = None) -> tuple[float, float, float]:
    """HAC t-stat of E[x] = 0. Returns (mean, t, two-sided p)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(x))
    se = newey_west_se(x, lags)
    if not np.isfinite(se) or se <= 0:
        return mu, float("nan"), float("nan")
    t = mu / se
    p = float(2.0 * stats.t.sf(abs(t), df=n - 1))
    return mu, float(t), p


@dataclass(frozen=True)
class DieboldMarianoResult:
    mean_loss_diff: float
    statistic: float
    p_value: float
    lags: int
    n: int
    preferred: str


def diebold_mariano(
    loss_a: Array,
    loss_b: Array,
    *,
    lags: int | None = None,
    name_a: str = "a",
    name_b: str = "b",
) -> DieboldMarianoResult:
    """DM test of equal predictive accuracy. Negative mean_loss_diff prefers a."""
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss series must align")
    d = a - b
    d = d[np.isfinite(d)]
    n = int(d.size)
    if lags is None:
        lags = max(1, int(np.floor(1.5 * n ** (1.0 / 3.0))))
    mu, t, p = mean_tstat(d, lags)
    if not np.isfinite(mu):
        pref = "inconclusive"
    elif mu < 0:
        pref = name_a
    elif mu > 0:
        pref = name_b
    else:
        pref = "tie"
    return DieboldMarianoResult(
        mean_loss_diff=mu,
        statistic=t,
        p_value=p,
        lags=int(lags),
        n=n,
        preferred=pref,
    )


def onesided_from_twosided(t: float, p_two: float, *, greater: bool = True) -> float:
    """Convert a two-sided t/z p-value to a one-sided p given the signed statistic."""
    if not np.isfinite(t) or not np.isfinite(p_two):
        return float("nan")
    half = 0.5 * float(p_two)
    if greater:
        return float(min(1.0, half if t > 0.0 else 1.0 - half))
    return float(min(1.0, half if t < 0.0 else 1.0 - half))


def mean_difference_t(
    mean_gap: float,
    n: int,
    *,
    sd: float | None = None,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """t-test of H0: E[gap] = 0 from a reported mean and n (df = n-1).

    Used when the date-level series is not in the family blob. If ``sd`` is
    omitted the scale is 1.0, so the test is conservative (low power) rather
    than a dummy ``p = 0 if better else 1`` map.
    """
    if n < 3 or not np.isfinite(mean_gap):
        return float("nan"), float("nan")
    scale = 1.0 if sd is None or not np.isfinite(sd) or sd <= 0.0 else float(sd)
    se = scale / float(np.sqrt(n))
    if se <= 0.0:
        return float("nan"), float("nan")
    t = float(mean_gap / se)
    df = int(n) - 1
    if alternative == "greater":
        p = float(stats.t.sf(t, df=df))
    elif alternative == "less":
        p = float(stats.t.cdf(t, df=df))
    else:
        p = float(2.0 * stats.t.sf(abs(t), df=df))
    return t, p


def two_proportion_test(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    *,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Test H0: p1 = p2. Returns (statistic, p_value).

    Pooled z when counts are large; Fisher exact on the 2×2 table otherwise.
    ``alternative='greater'`` is p1 > p2.
    """
    n1_i, n2_i = int(n1), int(n2)
    if n1_i < 1 or n2_i < 1:
        return float("nan"), float("nan")
    k1_i = min(max(int(k1), 0), n1_i)
    k2_i = min(max(int(k2), 0), n2_i)
    table = np.array([[k1_i, n1_i - k1_i], [k2_i, n2_i - k2_i]], dtype=int)
    p1 = k1_i / n1_i
    p2 = k2_i / n2_i
    pooled = (k1_i + k2_i) / (n1_i + n2_i)
    se = float(np.sqrt(max(pooled * (1.0 - pooled) * (1.0 / n1_i + 1.0 / n2_i), 0.0)))
    z = float((p1 - p2) / se) if se > 0.0 else 0.0
    sparse = min(n1_i, n2_i) < 20 or int(np.min(table)) < 5 or se <= 0.0
    fisher_alt = {
        "two-sided": "two-sided",
        "greater": "greater",
        "less": "less",
    }.get(alternative, "two-sided")
    if sparse:
        _odds, p = stats.fisher_exact(table, alternative=fisher_alt)
        return z if se > 0.0 else float(_odds), float(p)
    if alternative == "greater":
        p = float(stats.norm.sf(z))
    elif alternative == "less":
        p = float(stats.norm.cdf(z))
    else:
        p = float(2.0 * stats.norm.sf(abs(z)))
    return z, p


def benjamini_hochberg(
    p_values: Array, alpha: float = 0.05
) -> tuple[NDArray[np.bool_], float]:
    """BH FDR control. Returns (reject mask in original order, adaptive threshold)."""
    p = np.asarray(p_values, dtype=float)
    m = int(p.size)
    if m == 0:
        return np.array([], dtype=bool), 0.0
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if not below.any():
        return reject, 0.0
    k = int(np.max(np.nonzero(below)[0]))
    cutoff = float(thresh[k])
    reject[order[: k + 1]] = True
    return reject, cutoff


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> NDArray[np.intp]:
    if block < 1:
        raise ValueError("block must be >= 1")
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx: list[int] = []
    for s in starts:
        for k in range(block):
            idx.append(int((s + k) % n))
    return np.asarray(idx[:n], dtype=np.intp)


def bootstrap_mean_ci(
    x: Array,
    *,
    n_boot: int = 2000,
    block: int | None = None,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Circular-block bootstrap percentile CI for the mean. Returns (lo, hi, mean)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    if block is None:
        block = max(2, int(np.round(n ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    stats_boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = circular_block_indices(n, block, rng)
        stats_boot[i] = float(np.mean(x[idx]))
    lo = float(np.quantile(stats_boot, alpha / 2.0))
    hi = float(np.quantile(stats_boot, 1.0 - alpha / 2.0))
    return lo, hi, float(np.mean(x))


def bootstrap_sharpe_ci(
    returns: Array,
    *,
    n_boot: int = 2000,
    block: int | None = None,
    periods: int = 252,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Circular-block bootstrap CI for annualized Sharpe. Returns (lo, hi, point)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 10:
        return float("nan"), float("nan"), float("nan")
    if block is None:
        block = max(5, int(np.round(n ** (1.0 / 3.0))))

    def _sr(sample: Array) -> float:
        sd = float(np.std(sample, ddof=1))
        if sd <= 0:
            return 0.0
        return float(np.mean(sample) / sd * np.sqrt(periods))

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = circular_block_indices(n, block, rng)
        boots[i] = _sr(r[idx])
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi, _sr(r)


def jobson_korkie_memmel(sr_a: float, sr_b: float, n: int, corr: float) -> tuple[float, float]:
    """Memmel (2003) correction of Jobson–Korkie Sharpe difference test."""
    if n < 10:
        return float("nan"), float("nan")
    theta = (
        (sr_a - sr_b)
        / np.sqrt(
            (1.0 / n)
            * (2.0 * (1.0 - corr) + 0.5 * (sr_a**2 + sr_b**2 - sr_a * sr_b * (1.0 + corr**2)))
        )
    )
    p = float(2.0 * stats.norm.sf(abs(float(theta))))
    return float(theta), p
