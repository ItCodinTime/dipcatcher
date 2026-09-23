"""Value-at-Risk backtesting suite.

References:
- Kupiec (1995): unconditional-coverage LR test (proportion of failures).
- Christoffersen (1998): independence and conditional-coverage LR tests.
- Haas (2001): TUFF — time until first failure.
- Basel Committee (1996): traffic-light zones for 99% VaR over 250 days.
- Berkowitz, Christoffersen & Pelletier (2011): censored-normal transform.
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
        math.log(max(p, 1e-300)) + (t - 1) * math.log(max(1 - p, 1e-300))
        - math.log(phat) - (t - 1) * math.log(max(1 - phat, 1e-300))
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
