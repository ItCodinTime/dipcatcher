"""Engle-Granger cointegration screening with multiple-testing control.

Pipeline: correlation pre-filter on first differences -> Engle-Granger
(1987) two-step residual ADF test per candidate pair -> Bonferroni and
Benjamini-Hochberg adjusted p-values -> ranked candidate table.

Honesty notes (also surfaced in the receipt):
- The statsmodels ``adfuller`` p-value applies the *ordinary* Dickey-Fuller
  surface to residuals from an *estimated* regression. That is liberal
  relative to the Engle-Granger-specific MacKinnon surface. The table
  therefore carries both ``adf_pvalue`` (approximate) and the conservative
  ``passes_eg_cv`` flag (tau < EG 5% critical value); treat the adjusted
  p-values as approximate and the CV flag as the strict gate.
- The BH/Bonferroni family is the set of pairs that *survive the
  correlation pre-filter*. Filtering is itself selection, so the adjusted
  p-values control error within the tested family only.

References: Engle, Granger (1987); MacKinnon (1991) EG critical values;
Benjamini, Hochberg (1995).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from statsmodels.tsa.stattools import adfuller

from quant_fund.research.pairs.hedge import ols_hedge_ratio
from quant_fund.research.pairs.spread import ou_half_life

Array = NDArray[np.float64]

# Asymptotic Engle-Granger residual-test critical values for the
# two-variable case with a constant in the cointegrating regression
# (MacKinnon 1991 response surface, as tabulated for the EG procedure).
EG_CV_1PCT = -3.90
EG_CV_5PCT = -3.34
EG_CV_10PCT = -3.05

MIN_OBS = 60

_SCREEN_SCHEMA = {
    "a": pl.Int64,
    "b": pl.Int64,
    "corr_diff": pl.Float64,
    "direction": pl.String,
    "hedge_ratio": pl.Float64,
    "intercept": pl.Float64,
    "adf_tau": pl.Float64,
    "adf_pvalue": pl.Float64,
    "adf_lag": pl.Int64,
    "half_life": pl.Float64,
    "p_bonferroni": pl.Float64,
    "p_bh": pl.Float64,
    "passes_eg_cv": pl.Boolean,
    "passes_bh": pl.Boolean,
}


def _log_prices(prices: Array) -> Array:
    m = np.asarray(prices, dtype=float)
    if m.ndim != 2 or m.shape[0] < MIN_OBS or m.shape[1] < 2:
        raise ValueError(f"prices must be a finite (t, n) matrix with t >= {MIN_OBS}")
    if not np.all(np.isfinite(m)) or np.any(m <= 0):
        raise ValueError("prices must be finite and strictly positive")
    return np.log(m)


def correlation_prefilter(log_prices: Array, min_corr: float = 0.5) -> tuple[Array, Array]:
    """Candidate pairs whose |Pearson corr of first differences| >= min_corr.

    Correlation is computed on differenced log prices (returns), not
    levels — levels correlation between I(1) series is spurious by
    construction. Returns ``(pairs (m,2) int, abs_corr (m,))`` sorted by
    descending |corr|.
    """
    m = np.asarray(log_prices, dtype=float)
    if m.ndim != 2 or m.shape[0] < MIN_OBS or m.shape[1] < 2:
        raise ValueError("log_prices must be a finite (t, n) matrix")
    if not np.all(np.isfinite(m)):
        raise ValueError("log_prices must be finite")
    rets = np.diff(m, axis=0)
    if np.any(np.std(rets, axis=0) <= 0.0):
        raise ValueError("log_prices has a constant-difference column")
    corr = np.corrcoef(rets.T)
    iu = np.triu_indices(m.shape[1], k=1)
    abs_corr = np.abs(corr[iu])
    mask = abs_corr >= float(min_corr)
    order = np.argsort(-abs_corr[mask])
    pairs = np.column_stack([iu[0][mask][order], iu[1][mask][order]]).astype(int)
    return pairs, abs_corr[mask][order]


@dataclass(frozen=True)
class EGResult:
    """Result of one Engle-Granger residual-ADF run."""

    adf_tau: float
    adf_pvalue: float
    adf_lag: int
    hedge_ratio: float
    intercept: float
    resid: Array


def engle_granger_adf(y: Array, x: Array) -> EGResult:
    """Engle-Granger (1987) test: OLS ``y ~ x``, ADF on the residual.

    The ADF regression carries no deterministic terms (the residual is
    already demeaned through the cointegrating regression's intercept);
    lag order is chosen by ``adfuller``'s AIC search. See module docstring
    for why ``adf_pvalue`` is liberal on estimated residuals.
    """
    yv = np.asarray(y, dtype=float).reshape(-1)
    xv = np.asarray(x, dtype=float).reshape(-1)
    alpha, beta = ols_hedge_ratio(yv, xv)
    resid = yv - alpha - beta * xv
    if float(np.std(resid)) <= 0.0:
        raise ValueError("degenerate residual: series are proportional")
    stat, pvalue, usedlag, *_rest = adfuller(
        resid, regression="n", autolag="AIC", result_object=False
    )
    return EGResult(
        adf_tau=float(stat),
        adf_pvalue=float(pvalue),
        adf_lag=int(usedlag),
        hedge_ratio=beta,
        intercept=alpha,
        resid=resid,
    )


def benjamini_hochberg(pvalues: Array) -> Array:
    """BH (1995) adjusted p-values (q-values), implemented directly.

    Monotone step-up: ``q_(i) = min_{k>=i} min(1, m * p_(k) / k)``.
    """
    p = np.asarray(pvalues, dtype=float).reshape(-1)
    if p.size == 0 or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("pvalues must be a nonempty finite array in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = math.inf
    for rank in range(m - 1, -1, -1):  # rank is 0-based; m/(rank+1)
        idx = int(order[rank])
        running = min(running, p[idx] * m / (rank + 1))
        adj[idx] = min(running, 1.0)
    return adj


def bonferroni(pvalues: Array) -> Array:
    """Bonferroni-adjusted p-values ``min(1, m * p)``."""
    p = np.asarray(pvalues, dtype=float).reshape(-1)
    if p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("pvalues must be a nonempty finite array")
    out: Array = np.minimum(p * p.size, 1.0)
    return out


def screen_pairs(
    prices: Array,
    *,
    min_corr: float = 0.5,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Full screen: pre-filter -> EG residual ADF (both directions) -> BH.

    Tests ``a ~ b`` and ``b ~ a`` per candidate pair (the EG procedure is
    not symmetric) and keeps the direction with the smaller p-value.
    Returns a polars frame sorted by ``p_bh`` (ties: |corr| desc), one row
    per tested pair. ``passes_bh`` uses ``alpha``; ``passes_eg_cv`` is the
    conservative tau < EG 5% critical-value gate.
    """
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be in (0, 1)")
    logp = _log_prices(prices)
    pairs, corrs = correlation_prefilter(logp, min_corr=min_corr)
    base_rows: list[dict[str, object]] = []
    pvals: list[float] = []
    for (i, j), c in zip(pairs.tolist(), corrs.tolist(), strict=True):
        try:
            fwd = engle_granger_adf(logp[:, i], logp[:, j])  # a ~ b
            rev = engle_granger_adf(logp[:, j], logp[:, i])  # b ~ a
        except ValueError:
            continue
        pick, direction = (fwd, "a_on_b") if fwd.adf_pvalue <= rev.adf_pvalue else (rev, "b_on_a")
        base_rows.append(
            {
                "a": int(i),
                "b": int(j),
                "corr_diff": float(c),
                "direction": direction,
                "hedge_ratio": pick.hedge_ratio,
                "intercept": pick.intercept,
                "adf_tau": pick.adf_tau,
                "adf_pvalue": pick.adf_pvalue,
                "adf_lag": pick.adf_lag,
                "half_life": ou_half_life(pick.resid),
            }
        )
        pvals.append(pick.adf_pvalue)
    if not base_rows:
        return pl.DataFrame(schema=_SCREEN_SCHEMA)
    p_bh = benjamini_hochberg(np.asarray(pvals))
    p_bonf = bonferroni(np.asarray(pvals))
    for row, bh, bonf in zip(base_rows, p_bh.tolist(), p_bonf.tolist(), strict=True):
        row["p_bonferroni"] = bonf
        row["p_bh"] = bh
        row["passes_eg_cv"] = bool(cast(float, row["adf_tau"]) < EG_CV_5PCT)
        row["passes_bh"] = bool(bh < alpha)
    frame = pl.DataFrame(base_rows, schema=_SCREEN_SCHEMA, orient="row")
    return frame.sort(["p_bh", "corr_diff"], descending=[False, True])
