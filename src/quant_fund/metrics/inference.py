"""PhD-level inference: HAC, Diebold–Mariano, bootstrap CIs, FDR.

All p-values are two-sided unless noted. Loss convention for DM is
user-supplied (smaller is better). See docs/MATH_SPEC.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

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
    # Clamp the kernel HAC estimate at 0 (it can be slightly negative with
    # negative autocovariances), but never floor it above 0: a constant series
    # has true variance 0 and must fail closed as NaN downstream, not mint an
    # astronomical t-stat from an artificial epsilon.
    omega = max(omega, 0.0)
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


def grouped_mean_tstat(
    values: Array,
    groups: Array,
    *,
    target: float = 0.0,
    lags: int | None = None,
) -> tuple[float, float, float, int]:
    """HAC test of group-aggregated means against ``target``.

    Each group (normally a date) contributes one observation, preventing
    correlated panel names from inflating the nominal sample size.
    Research-diagnostic only.
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    if x.size != g.size:
        raise ValueError("values and groups must have the same length")
    mask = np.isfinite(x)
    x, g = x[mask], g[mask]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    _unique, inverse = np.unique(g, return_inverse=True)
    means = np.asarray(
        [float(np.mean(x[inverse == i])) for i in range(int(_unique.size))], dtype=float
    )
    mu, t, p = mean_tstat(means - float(target), lags=lags)
    return mu + float(target), t, p, int(means.size)


def two_way_clustered_mean_tstat(
    values: Array,
    cluster_a: Array,
    cluster_b: Array,
) -> tuple[float, float, float, int, int, int]:
    """Cameron–Gelbach–Miller two-way clustered t for E[x] = 0.

    Sandwich of the mean: ``V = V_a + V_b − V_white`` on demeaned residuals.
    Degrees of freedom are ``min(G_a, G_b) − 1``. Returns
    ``(mean, t, p, n, n_a, n_b)``. Degenerate when either way has < 2 clusters.
    """
    y = np.asarray(values, dtype=float).reshape(-1)
    a = np.asarray(cluster_a).reshape(-1)
    b = np.asarray(cluster_b).reshape(-1)
    if y.size != a.size or y.size != b.size:
        raise ValueError("values and cluster ids must align")
    mask = np.isfinite(y)
    y, a, b = y[mask], a[mask], b[mask]
    n = int(y.size)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n, 0, 0
    mu = float(np.mean(y))
    resid = y - mu

    def _cluster_var(groups: Array) -> tuple[float, int]:
        uniq, inv = np.unique(groups, return_inverse=True)
        g = int(uniq.size)
        if g < 2:
            return float("nan"), g
        sums = np.zeros(g, dtype=float)
        np.add.at(sums, inv, resid)
        return float(np.dot(sums, sums) / (n * n)), g

    v_white = float(np.dot(resid, resid) / (n * n))
    v_a, n_a = _cluster_var(a)
    v_b, n_b = _cluster_var(b)
    if not np.isfinite(v_a) or not np.isfinite(v_b) or n_a < 2 or n_b < 2:
        return mu, float("nan"), float("nan"), n, n_a, n_b
    var = v_a + v_b - v_white
    if var <= 0.0:
        return mu, float("nan"), float("nan"), n, n_a, n_b
    t_stat = mu / float(np.sqrt(var))
    df = max(1, min(n_a, n_b) - 1)
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df))
    return mu, float(t_stat), p_value, n, n_a, n_b


def wild_cluster_bootstrap_two_way_p(
    values: Array,
    cluster_a: Array,
    cluster_b: Array,
    *,
    observed_t: float,
    n_boot: int = 199,
    seed: int = 0,
) -> float:
    """Restricted Rademacher wild-cluster p for two-way clustered E[x] = 0.

    Wild weights hit ``cluster_a`` (dates in the sweep path). Under the null the
    residual is ``x`` itself. Few-cluster CGM t-tests over-reject; this p-value
    is the honesty companion, not a second discovery claim.
    """
    if not np.isfinite(observed_t) or int(n_boot) < 20:
        return float("nan")
    y = np.asarray(values, dtype=float).reshape(-1)
    a = np.asarray(cluster_a).reshape(-1)
    b = np.asarray(cluster_b).reshape(-1)
    if y.size != a.size or y.size != b.size:
        raise ValueError("values and cluster ids must align")
    mask = np.isfinite(y)
    y, a, b = y[mask], a[mask], b[mask]
    uniq, inv = np.unique(a, return_inverse=True)
    g = int(uniq.size)
    if g < 2 or int(y.size) < 3:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    rademacher = np.array([-1.0, 1.0])
    exceed = 0
    boots = int(n_boot)
    for _ in range(boots):
        weights = rng.choice(rademacher, size=g)
        _mu, t_star, _p, _n, _na, _nb = two_way_clustered_mean_tstat(y * weights[inv], a, b)
        if np.isfinite(t_star) and abs(float(t_star)) >= abs(float(observed_t)):
            exceed += 1
    return (1.0 + float(exceed)) / (float(boots) + 1.0)


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
    than a dummy ``p = 0 if better else 1`` map. An explicitly supplied
    non-finite or non-positive scale is invalid evidence and returns NaNs.
    """
    if n < 3 or not np.isfinite(mean_gap):
        return float("nan"), float("nan")
    if sd is None:
        scale = 1.0
    elif not np.isfinite(sd) or sd <= 0.0:
        return float("nan"), float("nan")
    else:
        scale = float(sd)
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
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
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    if n1_i < 1 or n2_i < 1 or not (0 <= int(k1) <= n1_i and 0 <= int(k2) <= n2_i):
        return float("nan"), float("nan")
    k1_i, k2_i = int(k1), int(k2)
    table = np.array([[k1_i, n1_i - k1_i], [k2_i, n2_i - k2_i]], dtype=int)
    p1 = k1_i / n1_i
    p2 = k2_i / n2_i
    pooled = (k1_i + k2_i) / (n1_i + n2_i)
    se = float(np.sqrt(max(pooled * (1.0 - pooled) * (1.0 / n1_i + 1.0 / n2_i), 0.0)))
    z = float((p1 - p2) / se) if se > 0.0 else 0.0
    sparse = min(n1_i, n2_i) < 20 or int(np.min(table)) < 5 or se <= 0.0
    fisher_alt = alternative
    if sparse:
        odds, p = stats.fisher_exact(table, alternative=fisher_alt)
        # Report the Fisher odds ratio with the Fisher p-value (a pooled z here
        # would not correspond to the exact test that produced p).
        return float(odds), float(p)
    if alternative == "greater":
        p = float(stats.norm.sf(z))
    elif alternative == "less":
        p = float(stats.norm.cdf(z))
    else:
        p = float(2.0 * stats.norm.sf(abs(z)))
    return z, p


def benjamini_hochberg(p_values: Array, alpha: float = 0.05) -> tuple[NDArray[np.bool_], float]:
    """BH FDR control. Returns (reject mask in original order, adaptive threshold)."""
    p = np.asarray(p_values, dtype=float)
    m = int(p.size)
    if m == 0:
        return np.array([], dtype=bool), 0.0
    if not np.isfinite(alpha) or not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and in (0, 1]")
    finite = np.isfinite(p)
    if np.any(finite & ((p < 0.0) | (p > 1.0))):
        raise ValueError("p_values must be in [0, 1] when finite")
    if not np.any(finite):
        return np.zeros(m, dtype=bool), 0.0
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
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([], dtype=np.intp)
    if block < 1:
        raise ValueError("block must be >= 1")
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx: list[int] = []
    for s in starts:
        for k in range(block):
            idx.append(int((s + k) % n))
    return np.asarray(idx[:n], dtype=np.intp)


def stationary_bootstrap_indices(
    n: int,
    n_boot: int,
    mean_block: float,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Stationary bootstrap (Politis & Romano 1994) index matrix, shape (n_boot, n).

    Block lengths are geometric with mean ``mean_block``: at each draw a new
    block starts (uniform index) with probability ``1/mean_block``, otherwise
    the current block continues, wrapping at the sample end. This keeps the
    resampled series strictly stationary.

    Fail-closed: ``n < 0`` / ``n_boot < 1`` / non-finite or out-of-range
    ``mean_block`` raise ValueError. ``n == 0`` returns an empty ``(n_boot, 0)``.
    """
    n_i = int(n)
    n_boot_i = int(n_boot)
    if n_i < 0:
        raise ValueError("n must be non-negative")
    if n_boot_i < 1:
        raise ValueError("n_boot must be >= 1")
    if not np.isfinite(mean_block):
        raise ValueError("mean_block must be finite")
    if n_i == 0:
        return np.empty((n_boot_i, 0), dtype=np.intp)
    if not 1.0 <= float(mean_block) <= float(n_i):
        raise ValueError("mean_block must be in [1, n]")
    p = 1.0 / float(mean_block)
    # Equivalent vectorized construction: an independent Bernoulli(p) "new block
    # starts here" indicator at every position gives geometric block lengths
    # (successes of a Bernoulli process), with a fresh uniform start per block.
    new_block = rng.random((n_boot_i, n_i)) < p
    new_block[:, 0] = True
    positions = np.arange(n_i)
    out = np.empty((n_boot_i, n_i), dtype=np.intp)
    for b in range(n_boot_i):
        starts = np.nonzero(new_block[b])[0]
        block_id = np.cumsum(new_block[b]) - 1
        origins = rng.integers(0, n_i, size=starts.size)
        out[b] = (origins[block_id] + positions - starts[block_id]) % n_i
    return out


def optimal_block_length(x: Array) -> float:
    """Politis–White (2004) automatic block length for the sample mean.

    Flat-top kernel with the ``m̂`` rule: the smallest ``m`` whose trailing
    ``K_N = max(10, ceil(sqrt(n)))`` autocorrelations all fall below
    ``2·sqrt(log10(n)/n)``; then ``M = 2m̂`` and

        b̂_opt = (2 Ĝ² / D̂²)^(1/3) · n^(1/3),

    with Ĝ = 2 Σ λ(k/M) k ρ̂(k) and D̂ = 1 + 2 Σ λ(k/M) ρ̂(k) over k = 1..M.
    Ĝ ≤ 0 / D̂ ≤ 0 (no detectable dependence) → 1.0; the result is clamped to
    ``[1, n-1]``. Constant series → 1.0; ``n < 10`` → NaN (fail-closed).

    Research diagnostic only — never a live Sharpe / P&L claim.
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 10:
        return float("nan")
    xc = arr - float(np.mean(arr))
    gamma0 = float(np.dot(xc, xc)) / n
    if gamma0 <= 0.0:
        return 1.0
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
    spec = np.fft.rfft(xc, nfft)
    acov = np.fft.irfft(spec * np.conj(spec), nfft)[:n] / n
    rho = acov / gamma0
    k_n = max(10, int(np.ceil(np.sqrt(n))))
    threshold = 2.0 * float(np.sqrt(np.log10(n) / n))
    max_m = max(1, n - 1 - k_n)
    m_hat = 1
    while m_hat <= max_m:
        tail = rho[m_hat + 1 : m_hat + 1 + k_n]
        if tail.size < k_n or bool(np.all(np.abs(tail) < threshold)):
            break
        m_hat += 1
    m_big = min(2 * m_hat, n - 1)
    lags = np.arange(1, m_big + 1, dtype=float)
    lam = np.where(lags <= m_big / 2.0, 1.0, 2.0 * (1.0 - lags / m_big))
    r = rho[1 : m_big + 1]
    g_hat = 2.0 * float(np.sum(lam * lags * r))
    d_hat = 1.0 + 2.0 * float(np.sum(lam * r))
    if g_hat <= 0.0 or d_hat <= 0.0:
        return 1.0
    b_opt = (2.0 * g_hat**2 / d_hat**2) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    if not np.isfinite(b_opt):
        return 1.0
    return float(min(max(b_opt, 1.0), max(1.0, n - 1.0)))


def bootstrap_mean_ci(
    x: Array,
    *,
    n_boot: int = 2000,
    block: int | None = None,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Circular-block bootstrap percentile CI for the mean. Returns (lo, hi, mean)."""
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if block is not None and int(block) < 1:
        raise ValueError("block must be >= 1")
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
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if block is not None and int(block) < 1:
        raise ValueError("block must be >= 1")
    if not np.isfinite(periods) or periods <= 0:
        raise ValueError("periods must be finite and positive")
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
    """Memmel (2003) correction of Jobson–Korkie Sharpe difference test.

    Research diagnostic only — not a live Sharpe / P&L claim.
    Fail-closed (nan, nan) on short n, non-finite inputs, or |corr| > 1.
    """
    if n < 10:
        return float("nan"), float("nan")
    if not (np.isfinite(sr_a) and np.isfinite(sr_b) and np.isfinite(corr)):
        return float("nan"), float("nan")
    if abs(float(corr)) > 1.0:
        return float("nan"), float("nan")
    # Memmel (2003): Var(SR_a − SR_b) =
    # (1/n)[2(1−ρ) + ½(SR_a² + SR_b² − 2ρ²·SR_a·SR_b)]
    denom = (1.0 / n) * (
        2.0 * (1.0 - corr) + 0.5 * (sr_a**2 + sr_b**2 - 2.0 * corr**2 * sr_a * sr_b)
    )
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan"), float("nan")
    theta = (sr_a - sr_b) / np.sqrt(denom)
    if not np.isfinite(theta):
        return float("nan"), float("nan")
    p = float(2.0 * stats.norm.sf(abs(float(theta))))
    return float(theta), p


def pairwise_diebold_mariano(
    loss_map: dict[str, Array],
    *,
    lags: int | None = None,
    include_e_process: bool = False,
    e_lam: float = 0.25,
    e_level: float = 0.05,
) -> list[dict[str, float | str | int | bool]]:
    """All pairwise Diebold–Mariano tests. ``loss_map`` values must align in length.

    Smaller loss is better. Returns one row per unordered pair with DM stats.
    Optional ``include_e_process`` adds research-only Choe–Ramdas-style capital
    fields (``e_final``, ``e_reject``, ``e_n``) for H0: E[L_a − L_b] ≤ 0 —
    never a live Sharpe / promotion claim.
    """
    from quant_fund.metrics.evalues import e_process_dm

    names = sorted(loss_map)
    rows: list[dict[str, float | str | int | bool]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            la = np.asarray(loss_map[a], dtype=float)
            lb = np.asarray(loss_map[b], dtype=float)
            if la.size != lb.size:
                # Fail closed on mismatched series: silent truncation would pair
                # observations from different times and corrupt the DM statistic.
                # Non-finite *elements* are handled downstream by diebold_mariano.
                raise ValueError(f"loss series must align: {a}={la.size}, {b}={lb.size}")
            n = la.size
            if n < 5:
                row: dict[str, float | str | int | bool] = {
                    "a": a,
                    "b": b,
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "preferred": "inconclusive",
                    "n": int(n),
                }
                if include_e_process:
                    row["e_final"] = float("nan")
                    row["e_reject"] = False
                    row["e_n"] = int(n)
                rows.append(row)
                continue
            dm = diebold_mariano(la, lb, lags=lags, name_a=a, name_b=b)
            row = {
                "a": a,
                "b": b,
                "statistic": float(dm.statistic),
                "p_value": float(dm.p_value),
                "preferred": dm.preferred,
                "mean_loss_diff": float(dm.mean_loss_diff),
                "n": int(dm.n),
            }
            if include_e_process:
                ep = e_process_dm(la, lb, lam=e_lam, level=e_level)
                row["e_final"] = cast(float, ep["e_final"])
                row["e_reject"] = cast(bool, ep["reject"])
                row["e_n"] = cast(int, ep["n"])
            rows.append(row)
    return rows
