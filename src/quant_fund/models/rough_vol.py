"""Rough-volatility estimators (Gatheral, Jaisson & Rosenbaum 2018).

Log-volatility behaves like fractional Brownian motion with H ~ 0.05-0.15;
this module implements the scaling diagnostics and the RFSV model.

References:
- Gatheral, Jaisson & Rosenbaum (2018): "Volatility is rough" — the
  q-th moment of Delta log-sigma scales as Delta^{qH}.
- Fukasawa (2011): asymptotic expansion of the RV estimator; related.
- Bennedsen, Lunde & Pakkanen (2017): hybrid scheme for fOU simulation.
- Bayer, Friz & Gatheral (2016): RFSV model specification.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _v(x: Array, n: int = 100) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {n}")
    return v


def logvol_hurst(
    log_vol: Array, qs: Array | None = None, max_lag: int = 50
) -> dict[str, Array | float]:
    """GJR (2018) moment-scaling estimate of H.

    For each lag Delta and moment q, ``m(q, Delta) = E|log sigma_{t+Delta}
    - log sigma_t|^q ~ Delta^{qH}``. Regressing log m on log Delta at
    fixed q gives slope zeta_q = qH; the estimate H = zeta_q / q.
    """
    v = _v(log_vol)
    qs_a = np.array([0.5, 1.0, 1.5, 2.0]) if qs is None else np.asarray(qs, dtype=float)
    if np.any(qs_a <= 0):
        raise ValueError("moments must be positive")
    n = v.size
    lags = np.arange(1, min(max_lag, n // 4) + 1)
    zetas = np.empty(qs_a.size)
    for qi, q in enumerate(qs_a):
        m = np.array([float(np.mean(np.abs(v[lag:] - v[:-lag]) ** q)) for lag in lags])
        good = m > 0
        if good.sum() < 5:
            raise ValueError("insufficient lag moments")
        slope = np.polyfit(np.log(lags[good]), np.log(m[good]), 1)[0]
        zetas[qi] = float(slope)
    h_est = zetas / qs_a
    return {
        "H": float(h_est.mean()),
        "H_by_q": h_est,
        "zeta_q": zetas,
        "qs": qs_a,
        "lags": lags.astype(float),
    }


def variance_curve_fit(log_vol: Array, lags: Array | None = None) -> dict[str, Array | float]:
    """Fit the fOU variance curve Var(log sigma_{t+D} - log sigma_t) =
    2 nu^2 D^{2H} (stationary increment form) by NLS on log-log.

    Returns (H, nu) and the fitted curve."""
    v = _v(log_vol)
    n = v.size
    lv = np.arange(1, min(30, n // 4) + 1) if lags is None else np.asarray(lags, dtype=float)
    lv = lv[(lv >= 1) & (lv < n // 3)]
    if lv.size < 5:
        raise ValueError("too few lags")
    var_d = np.array([float(np.var(v[int(li) :] - v[: -int(li)])) for li in lv])
    good = var_d > 0
    if good.sum() < 5:
        raise ValueError("degenerate variance curve")
    lg = np.log(lv[good])
    lv = np.log(var_d[good])
    # log var = log(2 nu^2) + 2H log D.
    A = np.column_stack([np.ones(good.sum()), 2.0 * lg])
    beta, *_ = np.linalg.lstsq(A, lv, rcond=None)
    H = float(beta[1])
    nu = math.sqrt(max(math.exp(beta[0]) / 2.0, 1e-20))
    return {
        "H": H,
        "nu": nu,
        "lags": lv[good],
        "var_observed": var_d[good],
        "var_fitted": np.exp(beta[0] + 2.0 * H * lg),
    }


def simulate_fou(
    n: int,
    H: float,
    nu: float = 1.0,
    theta: float = 0.0,
    seed: int = 0,
) -> Array:
    """Fractional Ornstein–Uhlenbeck path via the approximate fOU
    recursion: dX = -k X dt + nu dW^H, simulated by convolving Gaussian
    innovations with the fOU kernel (k->0 limit is pure fBM increments).

    Uses the Bennedsen-style power kernel W^H convolution truncated at
    n terms — adequate for testing estimators, not for pathwise claims."""
    if n < 50 or not (0.0 < H < 0.5):
        raise ValueError("need n >= 50 and H in (0, 0.5)")
    if nu <= 0:
        raise ValueError("nu must be positive")
    rng = np.random.default_rng(seed)
    # Fractional integration kernel for W^H increments:
    # w_k ~ k^{H - 1/2} (discretized power kernel).
    m = 2 * n
    w = np.arange(1, m + 1, dtype=float) ** (H - 0.5)
    w /= np.sqrt((w**2).sum() / m)  # normalize to unit variance scale
    e = rng.normal(size=m)
    conv = np.convolve(e, w)[:m]  # causal convolution
    x = conv[m - n :] * nu
    return np.asarray(x - x.mean() + theta)


def rough_signature(log_vol: Array, n_lags: int = 20) -> dict[str, Array | float]:
    """Rough-vol signature plot: lag-Delta std of Delta log-sigma vs Delta
    on log-log axes, returning the fitted roughness exponent H_hat/2 of
    the *standard deviation* scaling (std ~ Delta^H)."""
    v = _v(log_vol)
    lags = np.arange(1, min(n_lags, v.size // 4) + 1)
    sd = np.array([float(np.std(v[lag:] - v[:-lag])) for lag in lags])
    good = sd > 0
    if good.sum() < 5:
        raise ValueError("degenerate signature")
    fit = np.polyfit(np.log(lags[good]), np.log(sd[good]), 1)
    return {
        "lags": lags[good].astype(float),
        "std": sd[good],
        "H": float(fit[0]),
        "intercept": float(fit[1]),
    }
