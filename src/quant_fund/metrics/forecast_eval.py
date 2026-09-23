"""Out-of-sample forecast evaluation tests beyond Diebold–Mariano.

``metrics/inference.py`` already provides the DM test; this module adds
the nested-model and finite-sample corrections that dominate the modern
forecast-comparison literature.

References:
- Clark & West (2007): adjusted MSPE difference for nested forecasts.
- Harvey, Leybourne & Newbold (1997): small-sample DM correction.
- Giacomini & White (2006): conditional predictive ability (Wald).
- Harvey, Leybourne & Newbold (1998): forecast encompassing (ENC).
- Giacomini & Rossi (2010): fluctuation test for time-varying skill.
- McCracken (2007): MSPE-adjusted variant.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def _paired(e1: Array, e2: Array, n: int = 30) -> tuple[Array, Array]:
    a = np.asarray(e1, dtype=float).reshape(-1)
    b = np.asarray(e2, dtype=float).reshape(-1)
    if a.size != b.size or a.size < n:
        raise ValueError(f"loss series must be equal length >= {n}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("loss series must be finite")
    return a, b


def _lrv(d: Array, lag: int) -> float:
    """Newey–West long-run variance of a loss-difference series."""
    d = d - d.mean()
    n = d.size
    lag = min(lag, n - 1)
    g = float(np.dot(d, d) / n)
    for h in range(1, lag + 1):
        w = 1.0 - h / (lag + 1.0)
        g += 2.0 * w * float(np.dot(d[h:], d[:-h]) / n)
    return max(g, 1e-20)


def hln_test(e1: Array, e2: Array, lag: int | None = None) -> dict[str, float]:
    """Harvey–Leybourne–Newbold (1997) small-sample-corrected DM test.

    H0: equal predictive ability (E[e1^2 - e2^2] = 0). The HLN statistic
    rescales DM by sqrt([n + 1 - 2h + h(h-1)/n] / n) and uses t_{n-1}
    critical values. Returns dict with statistic/pvalue.
    """
    a, b = _paired(e1, e2)
    n = a.size
    h = 1 if lag is None else lag
    d = a**2 - b**2
    dm = float(d.mean() / math.sqrt(_lrv(d, max(0, h - 1)) / n))
    adj = math.sqrt((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n)
    stat = dm * adj
    p = 2.0 * (1.0 - float(stats.t.cdf(abs(stat), df=n - 1)))
    return {"statistic": stat, "pvalue": p, "dm": dm, "n": float(n)}


def clark_west_test(
    errors_alt: Array,
    errors_null: Array,
    preds_alt: Array,
) -> dict[str, float]:
    """Clark–West (2007) adjusted-MSPE test for *nested* comparisons.

    For nested models raw MSPE differences are biased toward the smaller
    null. The CW-adjusted loss difference is
    ``d_t = e_null,t^2 - e_alt,t^2 + (f_null,t - f_alt,t)^2``; a positive
    mean indicates the larger model beats the benchmark.

    Parameters
    ----------
    errors_alt : squared-error residuals of the larger (alternative) model.
    errors_null : squared-error residuals of the nested benchmark.
    preds_alt : forecast difference ``f_null - f_alt`` per period."""
    e2 = np.asarray(errors_alt, dtype=float).reshape(-1)
    e1 = np.asarray(errors_null, dtype=float).reshape(-1)
    fd = np.asarray(preds_alt, dtype=float).reshape(-1)
    if not (e1.size == e2.size == fd.size) or e1.size < 30:
        raise ValueError("series must share length >= 30")
    if not (np.all(np.isfinite(e1)) and np.all(np.isfinite(e2)) and np.all(np.isfinite(fd))):
        raise ValueError("series must be finite")
    n = e1.size
    d = e1**2 - e2**2 + fd**2
    stat = float(d.mean() / math.sqrt(_lrv(d, 0) / n))
    # One-sided p (H1: larger model better).
    p = 1.0 - float(stats.norm.cdf(stat))
    return {"statistic": stat, "pvalue": p, "mspe_adj": float(d.mean()), "n": float(n)}


def giacomini_white_test(
    loss1: Array, loss2: Array, instruments: Array | None = None
) -> dict[str, float]:
    """Giacomini–White (2006) unconditional predictive-ability Wald test.

    Regresses the loss difference on a constant plus optional instruments
    h_t (e.g., lagged loss diff); tests all coefficients = 0 via a Wald
    statistic with NW covariance. H0: equal conditional predictive ability.
    """
    l1, l2 = _paired(loss1, loss2)
    n = l1.size
    d = l1 - l2
    if instruments is None:
        X = np.column_stack([np.ones(n), np.concatenate([[0.0], d[:-1]])])
    else:
        Z = np.asarray(instruments, dtype=float)
        if Z.ndim != 2 or Z.shape[0] != n or not np.all(np.isfinite(Z)):
            raise ValueError("instruments must be finite (n, m)")
        X = np.column_stack([np.ones(n), Z])
    k = X.shape[1]
    beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    u = d - X @ beta
    # HAC covariance of OLS: (X'X)^{-1} (sum_t u_t^2 x_t x_t' + NW terms) (X'X)^{-1}.
    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError("instrument matrix is singular") from exc
    meat = np.zeros((k, k))
    ux = u[:, None] * X
    lag = int(math.floor(4.0 * (n / 100.0) ** 0.25))
    meat += ux.T @ ux
    for h in range(1, lag + 1):
        w = 1.0 - h / (lag + 1.0)
        meat += w * (ux[h:].T @ ux[:-h] + ux[:-h].T @ ux[h:])
    cov = xtx_inv @ meat @ xtx_inv
    try:
        w_stat = float(beta @ np.linalg.inv(cov) @ beta)
    except np.linalg.LinAlgError:
        w_stat = float("nan")
    if not np.isfinite(w_stat):
        raise ValueError("GW Wald statistic is degenerate")
    p = 1.0 - float(stats.chi2.cdf(w_stat, df=k))
    return {"statistic": w_stat, "pvalue": p, "df": float(k), "n": float(n)}


def encompassing_test(e_small: Array, e_large: Array, lag: int = 0) -> dict[str, float]:
    """Harvey–Leybourne–Newbold (1998) forecast-encompassing test (ENC-T).

    Tests whether the benchmark (small) forecast encompasses the rival:
    d_t = (e_small - e_large) * e_small; H0 E[d]=0 vs H1 > 0 (rival adds
    information). Positive stat => rival forecast adds value.
    """
    es, el = _paired(e_small, e_large)
    n = es.size
    d = (es - el) * es
    stat = float(d.mean() / math.sqrt(_lrv(d, lag) / n))
    p = 1.0 - float(stats.norm.cdf(stat))
    return {"statistic": stat, "pvalue": p, "n": float(n)}


def fluctuation_test(
    e1: Array,
    e2: Array,
    window: int,
    alpha: float = 0.10,
) -> dict[str, Array | float]:
    """Giacomini–Rossi (2010) fluctuation test for unstable relative skill.

    Computes rolling DM-type statistics over all windows of length
    ``window``; relative performance is flagged as unstable if any window
    statistic exceeds the GR critical value (asymptotic ~ +-3.18 at 10%
    for m/T -> 0, scaled here by sqrt(window)/... we use the sup-of-|stat|
    with the standard 10% asymptotic bound ~ 3.18/sqrt(1-2*log(...)) —
    in practice we report the sup and let the caller compare).
    """
    a, b = _paired(e1, e2)
    n = a.size
    if not (5 <= window <= n):
        raise ValueError("window must be in [5, n]")
    d = a**2 - b**2
    m = n - window + 1
    stats_roll = np.empty(m)
    for i in range(m):
        seg = d[i : i + window]
        v = _lrv(seg, 0)
        stats_roll[i] = seg.mean() / math.sqrt(v / window)
    sup = float(np.max(np.abs(stats_roll)))
    # GR (2010) asymptotic critical values for two-sided test, m->inf:
    # 10% ~ 3.18, 5% ~ 3.68 approx for the fluctuation statistic.
    cv = {0.10: 3.18, 0.05: 3.68}.get(alpha, 3.18)
    reject = float(sup > cv)
    return {
        "stats": stats_roll,
        "sup": sup,
        "critical": float(cv),
        "reject": reject,
        "argmax": float(np.argmax(np.abs(stats_roll))),
        "n": float(n),
    }
