"""Calibration and scoring extensions.

Extends the repo's existing proper-score suite (Brier/PIT/ECE in
``metrics/proper_scores.py`` and conformal in ``metrics/conformal.py``)
with decomposition and interval-scoring tools.

References:
- Murphy (1973) Brier-score decomposition: REL - RES + UNC.
- Winkler (1972) interval score; Gneiting, Raftery (2007) scoring rules.
- Dawid, Sebastiani (1999) coherent scoring on moments.
- Hosmer, Lemeshow (1980) goodness-of-fit for probabilistic classifiers.
- Murphy, Winkler (1977) reliability diagram binning.
- Scheuerer, Hamill (2015) variogram score for multivariate calibration.
- Candille, Talagrand (2005) binned spread/skill diagnostics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

Array = NDArray[np.float64]


def _v(x: Array, n: int = 5) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"input must be finite with length >= {n}")
    return v


def murphy_decomposition(
    prob_forecast: Array, outcome: Array, n_bins: int = 10
) -> dict[str, float]:
    """Murphy (1973) Brier decomposition: BS = REL - RES + UNC.

    Bins the probability forecasts; reliability is within-bin squared error,
    resolution is between-bin deviation from the climatology, uncertainty
    is the base-rate variance ``obar(1-obar)``.
    """
    p = _v(prob_forecast)
    y = _v(outcome)
    if p.size != y.size or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must match outcomes and lie in [0,1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("outcomes must be binary")
    if n_bins < 2:
        raise ValueError("n_bins >= 2")
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    obar = float(y.mean())
    rel = res = 0.0
    n = y.size
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        n_b = int(mask.sum())
        p_b = float(p[mask].mean())
        o_b = float(y[mask].mean())
        rel += (n_b / n) * (p_b - o_b) ** 2
        res += (n_b / n) * (o_b - obar) ** 2
    unc = obar * (1.0 - obar)
    bs = float(np.mean((p - y) ** 2))
    return {
        "brier": bs,
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "decomp_error": bs - (rel - res + unc),
    }


def winkler_interval_score(
    lower: Array, upper: Array, realized: Array, alpha: float = 0.1
) -> dict[str, Array]:
    """Winkler (1972)/Gneiting–Raftery interval score for (1-alpha) PIs.

    ``IS = (u - l) + (2/alpha)(l - y)1[y<l] + (2/alpha)(y - u)1[y>u]``.
    Lower is better; decomposes into width + penalties.
    """
    lo = _v(lower)
    hi = _v(upper)
    y = _v(realized)
    if lo.size != hi.size or lo.size != y.size:
        raise ValueError("lower/upper/realized must match")
    if np.any(hi < lo):
        raise ValueError("upper must exceed lower")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    width = hi - lo
    pen_lo = (2.0 / alpha) * (lo - y) * (y < lo)
    pen_hi = (2.0 / alpha) * (y - hi) * (y > hi)
    is_ = width + pen_lo + pen_hi
    return {
        "score": is_,
        "mean_score": np.array([float(is_.mean())]),
        "coverage": np.array([float(np.mean((y >= lo) & (y <= hi)))]),
        "mean_width": np.array([float(width.mean())]),
    }


def dawid_sebastiani(mean_fc: Array, var_fc: Array, realized: Array) -> dict[str, Array]:
    """Dawid–Sebastiani (1999) score: ``log(var) + (y - mu)^2 / var``.

    A proper score for forecasts summarized by mean and variance.
    """
    mu = _v(mean_fc)
    var = _v(var_fc)
    y = _v(realized)
    if mu.size != var.size or mu.size != y.size:
        raise ValueError("inputs must match")
    if np.any(var <= 0):
        raise ValueError("var must be positive")
    s = np.log(var) + (y - mu) ** 2 / var
    return {"score": s, "mean_score": np.array([float(s.mean())])}


def hosmer_lemeshow(prob_forecast: Array, outcome: Array, n_groups: int = 10) -> dict[str, float]:
    """Hosmer–Lemeshow (1980) decile calibration chi-squared test."""
    p = _v(prob_forecast)
    y = _v(outcome)
    if p.size != y.size or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must match outcomes in [0,1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("outcomes must be binary")
    if n_groups < 2 or n_groups > p.size // 5:
        raise ValueError("n_groups out of range")
    order = np.argsort(p)
    groups = np.array_split(order, n_groups)
    hl = 0.0
    for g in groups:
        o1 = float(y[g].sum())
        e1 = float(p[g].sum())
        n_g = float(g.size)
        e0 = n_g - e1
        o0 = n_g - o1
        if e1 > 0 and e0 > 0:
            hl += (o1 - e1) ** 2 / e1 + (o0 - e0) ** 2 / e0
    df = n_groups - 2
    return {"hl": hl, "pvalue": float(sstats.chi2.sf(hl, df)), "df": float(df)}


def reliability_diagram(prob_forecast: Array, outcome: Array, n_bins: int = 10) -> dict[str, Array]:
    """Murphy–Winkler reliability diagram data: bin centers, observed
    frequencies, and counts — the raw material for calibration curves.
    """
    p = _v(prob_forecast)
    y = _v(outcome)
    if p.size != y.size or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must match outcomes in [0,1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("outcomes must be binary")
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    centers = (np.arange(n_bins) + 0.5) / n_bins
    obs = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins)
    pbar = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bins == b
        if np.any(mask):
            obs[b] = float(y[mask].mean())
            pbar[b] = float(p[mask].mean())
            cnt[b] = float(mask.sum())
    return {"bin_center": centers, "obs_freq": obs, "mean_pred": pbar, "count": cnt}


def variogram_score(forecast_ens: Array, realized: Array, p: float = 0.5) -> float:
    """Scheuerer–Hamill (2015) variogram score for ensemble forecasts.

    ``VS = sum_ij w_ij |y_i - y_j|^p * |y_i - y_j|^p ...``
    concretely ``sum_ij ( |y_i - y_j|^p - E|X_i - X_j|^p )^2`` where X are
    ensemble draws and y the realized vector. Proper multivariate score.
    """
    ens = np.asarray(forecast_ens, dtype=float)
    y = np.asarray(realized, dtype=float).reshape(-1)
    if ens.ndim != 2 or not np.all(np.isfinite(ens)):
        raise ValueError("forecast_ens must be a finite (n_ens, d) matrix")
    if ens.shape[1] != y.size or not np.all(np.isfinite(y)):
        raise ValueError("realized must be finite length d matching ens cols")
    if p <= 0 or not np.isfinite(p):
        raise ValueError("p must be positive")
    dy = np.abs(y[:, None] - y[None, :]) ** p
    dxx = np.abs(ens[:, :, None] - ens[:, None, :]) ** p  # (m, d, d)
    exx = dxx.mean(axis=0)
    vs = float(np.sum((exx - dy) ** 2))
    return vs


def pinball_score(quantile_fc: Array, realized: Array, quantiles: Array) -> dict[str, Array]:
    """Pinball/check loss per quantile — the canonical quantile score.

    ``PS_tau = (tau - 1[y < q]) * (y - q)``; mean over obs per quantile.
    """
    q = np.asarray(quantile_fc, dtype=float)
    y = _v(realized)
    taus = np.asarray(quantiles, dtype=float).reshape(-1)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    if q.shape[0] != y.size or q.shape[1] != taus.size:
        raise ValueError("quantile_fc must be (n, n_tau) matching realized")
    if not np.all(np.isfinite(q)) or np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("non-finite forecasts or quantiles outside (0,1)")
    out = np.zeros(taus.size)
    for j, tau in enumerate(taus):
        diff = y - q[:, j]
        out[j] = float(np.mean(np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)))
    return {"per_quantile": out, "mean": np.array([float(out.mean())])}


def spread_skill(
    ens_mean: Array, ens_spread: Array, realized: Array, n_bins: int = 5
) -> dict[str, Array]:
    """Candille–Talagrand (2005) ensemble spread/skill diagnostic.

    Bins observations by ensemble spread; returns bin RMSE vs mean spread
    — a calibrated ensemble has RMSE tracking spread (slope ~ 1).
    """
    mu = _v(ens_mean)
    sd = _v(ens_spread)
    y = _v(realized)
    if mu.size != sd.size or mu.size != y.size or np.any(sd < 0):
        raise ValueError("inputs must match; spread non-negative")
    qs = np.quantile(sd, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    bin_rmse = np.full(n_bins, np.nan)
    bin_spread = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (sd >= qs[b]) & (sd < qs[b + 1])
        if np.any(mask):
            bin_rmse[b] = float(np.sqrt(np.mean((y[mask] - mu[mask]) ** 2)))
            bin_spread[b] = float(sd[mask].mean())
    valid = np.isfinite(bin_rmse)
    slope = (
        float(np.polyfit(bin_spread[valid], bin_rmse[valid], 1)[0]) if valid.sum() >= 2 else np.nan
    )
    return {
        "bin_rmse": bin_rmse,
        "bin_spread": bin_spread,
        "slope": np.array([slope]),
    }
