"""Value-at-Risk backtesting suite.

References:
- Kupiec (1995): unconditional-coverage LR test (proportion of failures).
- Christoffersen (1998): independence and conditional-coverage LR tests.
- Haas (2001): TUFF — time until first failure.
- Basel Committee (1996): traffic-light zones for 99% VaR over 250 days.
- Berkowitz, Christoffersen & Pelletier (2011): censored-normal transform.
- Kratz, Lok & McNeil (2018): multinomial test across nested VaR levels —
  evaluates tail shape, not just the violation rate at one level.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def _hits(hits: Array, n: int = 30) -> Array:
    h = np.asarray(hits, dtype=float).reshape(-1)
    if h.size < n or not np.all(np.isfinite(h)):
        raise ValueError(f"hit series must be finite with length >= {n}")
    if not np.all((h == 0.0) | (h == 1.0)):
        raise ValueError("hits must be binary 0/1")
    return h


def kupiec_test(hits: Array, alpha: float = 0.99) -> dict[str, float]:
    """Kupiec (1995) proportion-of-failures test.

    LR_uc = -2 ln[ (1-a)^{n-x} a^x / ((1-x/n)^{n-x} (x/n)^x) ] ~ chi2(1).
    H0: violation rate equals 1-alpha."""
    h = _hits(hits)
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha should be a tail level in (0.5, 1)")
    n = h.size
    x = int(h.sum())
    p = 1.0 - alpha
    phat = x / n
    if x == 0:
        lr = -2.0 * n * math.log(1.0 - p)
    else:
        lr = -2.0 * (
            (n - x) * math.log((1 - p) / max(1 - phat, 1e-300))
            + x * math.log(p / max(phat, 1e-300))
        )
    lr = float(max(lr, 0.0))
    return {
        "statistic": lr,
        "pvalue": float(1.0 - stats.chi2.cdf(lr, 1)),
        "failures": float(x),
        "expected": float(n * p),
        "rate": float(phat),
    }


def christoffersen_test(hits: Array, alpha: float = 0.99) -> dict[str, float]:
    """Christoffersen (1998) independence + conditional-coverage tests.

    Independence LR on the Markov transition matrix of hits; CC combines
    it with Kupiec's UC test (chi2(2)).
    """
    h = _hits(hits)
    n = h.size
    n00 = int(np.sum((h[:-1] == 0) & (h[1:] == 0)))
    n01 = int(np.sum((h[:-1] == 0) & (h[1:] == 1)))
    n10 = int(np.sum((h[:-1] == 1) & (h[1:] == 0)))
    n11 = int(np.sum((h[:-1] == 1) & (h[1:] == 1)))
    if n10 + n11 == 0 or n00 + n01 == 0:
        raise ValueError("no transitions in one state — independence test undefined")
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(p01: float, p11: float) -> float:
        out = 0.0
        for cnt, pr in ((n00, 1.0 - p01), (n01, p01), (n10, 1.0 - p11), (n11, p11)):
            if cnt > 0:
                out += cnt * math.log(max(pr, 1e-300))
        return out

    ll_unres = ll(pi01, pi11)
    ll_res = ll(pi, pi)
    lr_ind = float(max(-2.0 * (ll_res - ll_unres), 0.0))
    p_ind = float(1.0 - stats.chi2.cdf(lr_ind, 1))
    uc = kupiec_test(h, alpha)
    lr_cc = float(uc["statistic"] + lr_ind)
    return {
        "lr_independence": lr_ind,
        "pvalue_independence": p_ind,
        "lr_cc": lr_cc,
        "pvalue_cc": float(1.0 - stats.chi2.cdf(lr_cc, 2)),
        "pi01": float(pi01),
        "pi11": float(pi11),
        "n": float(n),
    }


def tuff_test(hits: Array, alpha: float = 0.99) -> dict[str, float]:
    """Haas (2001) time-until-first-failure test.

    Under correct coverage, time-to-first-hit is Geometric(p=1-alpha);
    the LR test compares to the geometric MLE."""
    h = _hits(hits)
    p = 1.0 - alpha
    first = int(np.argmax(h > 0.5)) if np.any(h > 0.5) else h.size
    # TUFF LR for time to first failure t:
    # LR = -2 ln[ p (1-p)^{t-1} / ( (1/t) (1 - 1/t)^{t-1} ) ]
    t = max(first + 1, 1)
    phat = 1.0 / t
    lr = -2.0 * (
        math.log(max(p, 1e-300))
        + (t - 1) * math.log(max(1 - p, 1e-300))
        - math.log(phat)
        - (t - 1) * math.log(max(1 - phat, 1e-300))
    )
    lr = float(max(lr, 0.0))
    return {
        "statistic": lr,
        "pvalue": float(1.0 - stats.chi2.cdf(lr, 1)),
        "time_to_first": float(t),
        "expected": float(1.0 / p),
    }


def basel_zone(hits: Array, alpha: float = 0.99) -> dict[str, float | str]:
    """Basel Committee (1996) traffic-light zones for VaR exceptions.

    Standard calibration: 99% VaR over ~250 trading days:
    green <= 4, yellow 5-9, red >= 10 exceptions. Generalized here via
    the exact binomial CDF cutoffs: green if P(X >= x) > 5% under H0,
    yellow down to P <= 0.01%, red beyond. For the canonical 99%/250d
    case the fixed table is used."""
    h = _hits(hits, n=10)
    x = int(h.sum())
    n = h.size
    p = 1.0 - alpha
    if abs(alpha - 0.99) < 1e-9 and n == 250:
        zone = 0 if x <= 4 else (1 if x <= 9 else 2)
        labels = ("green", "yellow", "red")
        return {"zone": float(zone), "label": labels[zone], "failures": float(x), "n": float(n)}
    # Generic: probability of >= x failures under H0.
    tail = float(stats.binom.sf(x - 1, n, p)) if x > 0 else 1.0
    zone = 0 if tail > 0.05 else (1 if tail > 0.0001 else 2)
    labels = ("green", "yellow", "red")
    return {
        "zone": float(zone),
        "label": labels[zone],
        "failures": float(x),
        "n": float(n),
        "tail_prob": tail,
    }


def kratz_test(returns: Array, var_levels: Array, alphas: Array) -> dict[str, float | Array]:
    """Kratz–Lok–McNeil (2018) multinomial tail test.

    Instead of one VaR level, takes forecasts at ``K`` nested levels
    ``alphas`` (strictly increasing, e.g. ``[0.95, 0.975, 0.99, 0.999]``).
    Each day lands in one of ``K+1`` bands between consecutive VaR
    forecasts; under correct specification the band probabilities are
    the fixed differences ``pi_j = alpha_{j+1} - alpha_j`` regardless of
    how the VaR moves in time, so band counts are multinomial and a
    Pearson X^2 (and LR) statistic is ~ chi2(K). Detects tail-shape
    misspecification that a single-level Kupiec test cannot.

    Loss convention like the rest of the module: ``r_t > var_t`` is an
    exceedance, so ``var_levels`` must be non-decreasing across columns.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    v = np.asarray(var_levels, dtype=float)
    a = np.asarray(alphas, dtype=float).reshape(-1)
    if r.size < 50:
        raise ValueError("returns must have length >= 50")
    if v.ndim != 2 or v.shape[0] != r.size or v.shape[1] < 2:
        raise ValueError("var_levels must be (n, K) with K >= 2")
    if a.size != v.shape[1]:
        raise ValueError("alphas must match var_levels columns")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v)) and np.all(np.isfinite(a))):
        raise ValueError("inputs must be finite")
    if np.any(a <= 0.0) or np.any(a >= 1.0) or np.any(np.diff(a) <= 0.0):
        raise ValueError("alphas must be strictly increasing in (0, 1)")
    if np.any(np.diff(v, axis=1) < 0.0):
        raise ValueError("var_levels must be non-decreasing across levels each day")
    k = a.size
    n = r.size
    # Band j: r_t in (var_{j-1}, var_j]; band 0 is (-inf, var_1],
    # band K is (var_K, +inf). Fixed probabilities under correct spec.
    edges = np.concatenate([np.full((n, 1), -np.inf), v, np.full((n, 1), np.inf)], axis=1)
    band = np.sum(r[:, None] > edges[:, 1:], axis=1)  # count of thresholds exceeded
    counts = np.bincount(band, minlength=k + 1).astype(float)
    pi = np.diff(np.concatenate([[0.0], a, [1.0]]))
    expected = n * pi
    if np.any(expected < 1.0):
        raise ValueError("expected band counts < 1 — raise n or drop extreme levels")
    x2 = float(np.sum((counts - expected) ** 2 / expected))
    nz = counts > 0
    g2 = float(2.0 * np.sum(counts[nz] * np.log(counts[nz] / expected[nz])))
    return {
        "statistic": x2,
        "lr": g2,
        "pvalue": float(1.0 - stats.chi2.cdf(x2, k)),
        "pvalue_lr": float(1.0 - stats.chi2.cdf(g2, k)),
        "df": float(k),
        "n": float(n),
        "counts": counts,
        "expected": expected,
    }


def _kupiec_lr(x: Array, n: int, p: float) -> Array:
    """Kupiec LR_uc as a function of the violation count x (vectorized)."""
    x = np.asarray(x, dtype=float)
    ll_null = (n - x) * np.log1p(-p) + x * np.log(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        phat = x / n
        ll_alt = np.where(
            (x > 0) & (x < n),
            (n - x) * np.log1p(-phat) + x * np.log(phat),
            np.where(x == 0, n * np.log1p(-0.0), n * np.log(1.0)),
        )
    return np.asarray(-2.0 * (ll_null - ll_alt), dtype=float)


def dumitrescu_hurlin_test(hits_panel: Array, alpha: float = 0.99) -> dict[str, float]:
    """Dumitrescu–Hurlin (2012) panel VaR coverage test.

    Pools per-series Kupiec LR_uc statistics across ``N`` series:
    ``Z = (mean_i LR_i - mean_i E_i) / sqrt(sum_i V_i / N^2) ~ N(0,1)``,
    where ``E_i``/``V_i`` are the exact binomial moments of the LR
    statistic under H0 for series length ``n_i`` — LR_uc is a pure
    function of the count ``x ~ Binomial(n_i, 1-alpha)``, so the
    correction is computed exactly rather than by simulation as in the
    original paper.

    ``hits_panel`` is (T, N): column i is series i's hit sequence
    (binary, finite). Series may have leading all-zero padding; each
    column's effective length is its full T — ragged panels should be
    pre-trimmed by the caller.
    """
    h = np.asarray(hits_panel, dtype=float)
    if h.ndim != 2 or h.shape[0] < 30 or h.shape[1] < 2:
        raise ValueError("hits_panel must be (T, N) with T >= 30, N >= 2")
    if not np.all(np.isfinite(h)) or not np.all((h == 0.0) | (h == 1.0)):
        raise ValueError("hits_panel must be finite binary 0/1")
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must be in (0.5, 1)")
    t, n_series = h.shape
    p = 1.0 - alpha
    lr_i = _kupiec_lr(h.sum(axis=0), t, p)
    # Exact moments of LR_uc under H0: x ~ Binomial(t, p).
    xs = np.arange(t + 1, dtype=float)
    pmf = stats.binom.pmf(xs, t, p)
    lr_x = _kupiec_lr(xs, t, p)
    e_lr = float(np.sum(pmf * lr_x))
    v_lr = float(np.sum(pmf * lr_x * lr_x) - e_lr * e_lr)
    if v_lr <= 0.0:
        raise ValueError("degenerate null distribution")
    z = float((lr_i.mean() - e_lr) / math.sqrt(v_lr / n_series))
    return {
        "statistic": z,
        "pvalue": float(2.0 * stats.norm.sf(abs(z))),
        "n_series": float(n_series),
        "t": float(t),
        "mean_lr": float(lr_i.mean()),
        "expected_lr": e_lr,
    }
