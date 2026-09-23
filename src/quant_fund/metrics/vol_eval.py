"""Volatility forecast evaluation.

- ``mincer_zarnowitz``: MZ regression realized_t = a + b * forecast_t
  with OLS estimates, a joint Wald test of (a=0, b=1), and R2.
- Patton (2011) loss functions for variance forecasts given a noisy
  proxy x (e.g., squared return or realized variance): ``mse``,
  ``qlike`` (robust to proxy noise), ``mse_log``, ``hmse``, ``mae``.
- ``vol_loss_diff``: mean loss differential between two forecasts with
  a Newey-West standard error (a DM-ready pairwise comparison).

Fail-closed: non-positive forecasts for log-based losses, non-finite
input, length mismatch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def _check(x: Array, f: Array) -> tuple[Array, Array]:
    xx = np.asarray(x, dtype=float).ravel()
    ff = np.asarray(f, dtype=float).ravel()
    if xx.size != ff.size or xx.size < 10:
        raise ValueError("x and f must match, >= 10 obs")
    if not np.isfinite(xx).all() or not np.isfinite(ff).all():
        raise ValueError("non-finite input")
    return xx, ff


def mincer_zarnowitz(x: Array, f: Array) -> dict[str, float]:
    """MZ efficiency regression x_t = a + b f_t + e_t.

    Returns a, b, their standard errors, joint Wald chi2(2) stat for
    H0: a=0 & b=1, its p-value, and R2.
    """
    xx, ff = _check(x, f)
    n = xx.size
    xmat = np.column_stack([np.ones(n), ff])
    coef, *_ = np.linalg.lstsq(xmat, xx, rcond=None)
    resid = xx - xmat @ coef
    s2 = float(resid @ resid / (n - 2))
    cov_b = s2 * np.linalg.inv(xmat.T @ xmat)
    se = np.sqrt(np.maximum(np.diag(cov_b), 0.0))
    # joint Wald: R coef = (0, 1)
    rvec = coef - np.array([0.0, 1.0])
    try:
        wald = float(rvec @ np.linalg.inv(cov_b) @ rvec)
    except np.linalg.LinAlgError:
        wald = np.nan
    p = float(1.0 - stats.chi2.cdf(wald, 2)) if np.isfinite(wald) else np.nan
    ss_res = float(resid @ resid)
    ss_tot = float(((xx - xx.mean()) ** 2).sum())
    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "se_alpha": float(se[0]),
        "se_beta": float(se[1]),
        "wald": wald,
        "wald_p": p,
        "r2": 1.0 - ss_res / max(ss_tot, 1e-14),
    }


def mse(x: Array, f: Array) -> Array:
    xx, ff = _check(x, f)
    return (xx - ff) ** 2


def qlike(x: Array, f: Array) -> Array:
    """Patton-robust QLIKE: x/f - ln(x/f) - 1 (needs x, f > 0)."""
    xx, ff = _check(x, f)
    if (ff <= 0).any() or (xx <= 0).any():
        raise ValueError("qlike requires positive forecasts and proxies")
    r = xx / ff
    return r - np.log(r) - 1.0


def mse_log(x: Array, f: Array) -> Array:
    xx, ff = _check(x, f)
    if (ff <= 0).any() or (xx <= 0).any():
        raise ValueError("mse_log requires positive inputs")
    return (np.log(xx) - np.log(ff)) ** 2


def hmse(x: Array, f: Array) -> Array:
    """Heteroskedasticity-adjusted MSE: (1 - x/f)^2 (Bollerslev-Ghysels)."""
    xx, ff = _check(x, f)
    if (ff <= 0).any():
        raise ValueError("hmse requires positive forecasts")
    return (1.0 - xx / ff) ** 2


def mae(x: Array, f: Array) -> Array:
    xx, ff = _check(x, f)
    return np.abs(xx - ff)


_LOSSES = {"mse": mse, "qlike": qlike, "mse_log": mse_log, "hmse": hmse, "mae": mae}


def vol_loss_diff(
    x: Array, f1: Array, f2: Array, loss: str = "qlike", max_lag: int | None = None
) -> dict[str, float]:
    """Mean loss differential d_t = L(x, f1) - L(x, f2) with NW se.

    Positive diff means f2 is better. Returns mean diff, se, t-stat.
    """
    if loss not in _LOSSES:
        raise ValueError(f"loss must be one of {sorted(_LOSSES)}")
    l1 = _LOSSES[loss](x, f1)
    l2 = _LOSSES[loss](x, f2)
    d = l1 - l2
    n = d.size
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    dc = d - d.mean()
    s = float(dc @ dc) / n
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        s += 2.0 * w * float(dc[lag:] @ dc[:-lag]) / n
    se = np.sqrt(max(s / n, 0.0))
    return {
        "mean_diff": float(d.mean()),
        "se": float(se),
        "t": float(d.mean() / max(se, 1e-14)),
        "n": float(n),
    }
