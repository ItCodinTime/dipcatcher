"""Unit-safe PSR / MinTRL — fixes the A1 F1 class of bug.

The audit finding A1 F1: ``hedge_lab/scoreboard.py`` fed an **annualized**
Sharpe (``mean/vol * sqrt(252)``) into ``probabilistic_sharpe`` together with a
per-day ``n_obs``, inflating the z-statistic by ``sqrt(252) ~ 15.9x`` and
short-circuiting ``psr_vs_zero`` to ~1.0 for any positive-Sharpe book.

The functions here make that mistake **unexpressible**: the API takes raw
returns, never a pre-computed Sharpe, so per-period moments are computed
internally with one uniform convention (Bailey & López de Prado: SR in the
same frequency as the observations; skew / raw kurtosis per Lo 2002).

Research diagnostics only — never a live P&L / promotion claim.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.overfitting import (
    min_track_record_length,
    moments_from_returns,
    probabilistic_sharpe,
)

Array = NDArray[np.float64]


def _clean_returns(returns: Array) -> Array:
    r = np.asarray(returns, dtype=float).reshape(-1)
    return r[np.isfinite(r)]


def _validate_periods_per_year(periods_per_year: float) -> None:
    if not np.isfinite(periods_per_year) or float(periods_per_year) <= 0.0:
        raise ValueError("periods_per_year must be finite and positive")


def _psr_payload(n_obs: int) -> dict[str, float]:
    nan = float("nan")
    return {
        "psr": nan,
        "sr_periodic": nan,
        "sr_annualized": nan,
        "n_obs": float(n_obs),
        "skew": nan,
        "kurtosis_raw": nan,
    }


def psr_from_returns(
    returns: Array,
    *,
    sr_star: float = 0.0,
    periods_per_year: float,
) -> dict[str, float]:
    """Unit-safe PSR: per-period moments computed INTERNALLY, then
    ``metrics.overfitting.probabilistic_sharpe`` with the SR in PERIOD units.

    Returns ``{'psr', 'sr_periodic', 'sr_annualized', 'n_obs', 'skew',
    'kurtosis_raw'}`` — the annualized value is present for display only and
    is never fed back into inference. Short (<4 finite obs), zero-vol, or
    non-finite paths are fail-closed NaN (never a fabricated 0.0 or 1.0).
    """
    _validate_periods_per_year(periods_per_year)
    r = _clean_returns(returns)
    n = int(r.size)
    out = _psr_payload(n)
    if n < 4:
        return out
    sig, skew, kurt = moments_from_returns(r)
    if not (np.isfinite(sig) and np.isfinite(skew) and np.isfinite(kurt)) or np.ptp(r) == 0.0:
        return out
    sr_periodic = float(np.mean(r) / sig)
    psr = probabilistic_sharpe(sr_periodic, float(sr_star), n, float(skew), float(kurt))
    out.update(
        psr=float(psr),
        sr_periodic=sr_periodic,
        sr_annualized=sr_periodic * float(np.sqrt(float(periods_per_year))),
        skew=float(skew),
        kurtosis_raw=float(kurt),
    )
    return out


def min_trl_from_returns(
    returns: Array,
    *,
    sr_star: float = 0.0,
    periods_per_year: float,
) -> float:
    """MinTRL in PERIODS via ``metrics.overfitting.min_track_record_length``
    on the per-period SR. Fail-closed ``float('nan')`` on short/non-finite
    input or when ``sr <= sr_star`` is not separable at the target confidence.
    """
    _validate_periods_per_year(periods_per_year)
    r = _clean_returns(returns)
    if int(r.size) < 4:
        return float("nan")
    sig, skew, kurt = moments_from_returns(r)
    if not (np.isfinite(sig) and np.isfinite(skew) and np.isfinite(kurt)) or np.ptp(r) == 0.0:
        return float("nan")
    sr_periodic = float(np.mean(r) / sig)
    if sr_periodic <= float(sr_star):
        # MinTRL is undefined when the observed SR does not exceed the
        # benchmark SR: fail-closed NaN (the underlying Bailey–LdP formula
        # would otherwise return a meaningless finite count).
        return float("nan")
    return min_track_record_length(sr_periodic, float(skew), float(kurt), sr_star=float(sr_star))
