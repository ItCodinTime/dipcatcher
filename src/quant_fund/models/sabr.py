"""SABR stochastic-volatility smile: implied vol, calibration, shifted SABR.

Hagan, Kumar, Lesniewski & Woodward (2002), "Managing smile risk",
Wilmott Magazine. Parameters: ``alpha`` (vol level), ``beta`` (CEV
elasticity, fixed at 0<=beta<=1 in calibration), ``rho`` (spot-vol
correlation), ``nu`` (vol-of-vol). The Hagan expansion is valid for
moderate strikes/maturities; it is not arbitrage-free in the wings.

Fail-closed: non-positive forward/strike/tenor/alpha/nu or |rho|>=1
raise ``ValueError``; non-finite inputs raise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

Array = NDArray[np.float64]


def _check_sabr_params(alpha: float, beta: float, rho: float, nu: float) -> None:
    if not np.isfinite([alpha, beta, rho, nu]).all():
        raise ValueError("sabr params must be finite")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if abs(rho) >= 1.0:
        raise ValueError("rho must be in (-1, 1)")
    if nu <= 0.0:
        raise ValueError("nu must be > 0")


def sabr_implied_vol(
    forward: float,
    strike: Array | float,
    tenor: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
) -> Array:
    """Hagan (2002) implied Black vol. ``shift``>0 gives shifted SABR
    (handles negative forwards/strikes: evaluated at f+shift, K+shift)."""
    _check_sabr_params(alpha, beta, rho, nu)
    if not np.isfinite(forward) or not np.isfinite(tenor) or tenor <= 0.0:
        raise ValueError("forward must be finite, tenor > 0")
    f = float(forward) + float(shift)
    k = np.asarray(strike, dtype=float) + float(shift)
    if not np.isfinite(k).all():
        raise ValueError("strikes must be finite")
    if f <= 0.0 or (k <= 0.0).any():
        raise ValueError("shifted forward/strikes must be > 0")

    fk = f * k
    one_m_b = 1.0 - beta
    log_fk = np.log(f / k)
    fk_beta = fk ** (one_m_b / 2.0)

    atm = np.abs(log_fk) < 1e-12

    # ATM limit (Hagan 2002 eq. A.67 with z -> 0 handled separately).
    sig_atm = (alpha / f**one_m_b) * (
        1.0
        + tenor
        * (
            (one_m_b**2 / 24.0) * alpha**2 / f ** (2.0 * one_m_b)
            + (rho * beta * nu * alpha) / (4.0 * f**one_m_b)
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        )
    )
    out = np.where(atm, sig_atm, 0.0)

    if (~atm).any():
        fk_m = fk[~atm]
        logm = log_fk[~atm]
        fk_bm = fk_beta[~atm]
        z = (nu / alpha) * fk_bm * logm
        xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))
        denom = fk_bm * (1.0 + (one_m_b**2 / 24.0) * logm**2 + (one_m_b**4 / 1920.0) * logm**4)
        corr = 1.0 + tenor * (
            (one_m_b**2 / 24.0) * alpha**2 / (fk_m**one_m_b)
            + (rho * beta * nu * alpha) / (4.0 * fk_bm)
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        )
        out[~atm] = (alpha / denom) * (z / xz) * corr
    return np.asarray(out, dtype=float)


def sabr_atm_vol(
    forward: float,
    tenor: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """ATM Black vol under SABR (limit of :func:`sabr_implied_vol`)."""
    return float(sabr_implied_vol(forward, forward, tenor, alpha, beta, rho, nu))


def sabr_alpha_from_atm(
    forward: float,
    tenor: float,
    atm_vol: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Solve the ATM equation for alpha given (beta, rho, nu, sigma_atm).

    The ATM formula is cubic in alpha; the unique positive root is found
    by bracketed bisection/Brent on [eps, hi].
    """
    _check_sabr_params(1.0, beta, rho, nu)
    if (
        not np.isfinite([forward, tenor, atm_vol]).all()
        or forward <= 0.0
        or tenor <= 0.0
        or atm_vol <= 0.0
    ):
        raise ValueError("forward, tenor, atm_vol must be finite and > 0")
    one_m_b = 1.0 - beta
    f = float(forward)
    target = f**one_m_b * atm_vol

    def g(a: float) -> float:
        return float(
            tenor * (one_m_b**2 / 24.0) / f ** (2.0 * one_m_b) * a**3
            + tenor * (rho * beta * nu) / (4.0 * f**one_m_b) * a**2
            + (1.0 + tenor * ((2.0 - 3.0 * rho**2) / 24.0) * nu**2) * a
            - target
        )

    lo, hi = 1e-12, max(1.0, 4.0 * target)
    while g(hi) < 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError("cannot bracket positive root for alpha")
    return float(optimize.brentq(g, lo, hi, xtol=1e-14))


def sabr_fit(
    forward: float,
    tenor: float,
    strikes: Array,
    market_vols: Array,
    beta: float = 0.5,
    fit_beta: bool = False,
    weights: Array | None = None,
) -> dict[str, float]:
    """Least-squares SABR smile calibration.

    Returns dict with alpha, beta, rho, nu, rmse. ``beta`` is fixed by
    default (standard practice: it is weakly identified with rho/nu);
    pass ``fit_beta=True`` to free it (bounded in [0, 1]).
    """
    k = np.asarray(strikes, dtype=float).ravel()
    v = np.asarray(market_vols, dtype=float).ravel()
    if k.shape != v.shape or k.size < 3:
        raise ValueError("strikes and market_vols must match, >= 3")
    if not np.isfinite(k).all() or not np.isfinite(v).all() or (k <= 0.0).any() or (v <= 0.0).any():
        raise ValueError("strikes and vols must be finite and > 0")
    if not np.isfinite(forward) or not np.isfinite(tenor) or forward <= 0.0 or tenor <= 0.0:
        raise ValueError("forward, tenor must be finite and > 0")
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape != v.shape or not np.isfinite(w).all() or (w < 0.0).any():
        raise ValueError("weights must match vols, finite, >= 0")

    atm_idx = int(np.argmin(np.abs(k - forward)))
    sigma_atm = float(v[atm_idx])

    def residuals(theta: Array, fixed_beta: bool) -> Array:
        if fixed_beta:
            alpha, rho, nu = float(theta[0]), float(theta[1]), float(theta[2])
            b = beta
        else:
            alpha, rho, nu, b = (
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
                float(theta[3]),
            )
        model = sabr_implied_vol(forward, k, tenor, alpha, b, rho, nu)
        return np.asarray((model - v) * w, dtype=float)

    if fit_beta:
        x0 = np.array([max(sigma_atm * forward ** (1.0 - beta), 1e-4), 0.0, 0.5, beta])
        bounds = ([1e-8, -0.999, 1e-8, 0.0], [np.inf, 0.999, np.inf, 1.0])
    else:
        try:
            a0 = sabr_alpha_from_atm(forward, tenor, sigma_atm, beta, 0.0, 0.5)
        except ValueError:
            a0 = max(sigma_atm * forward ** (1.0 - beta), 1e-4)
        x0 = np.array([a0, 0.0, 0.5])
        bounds = ([1e-8, -0.999, 1e-8], [np.inf, 0.999, np.inf])

    res = optimize.least_squares(residuals, x0, bounds=bounds, args=(not fit_beta,), method="trf")
    if fit_beta:
        alpha, rho, nu, b_hat = map(float, res.x)
    else:
        alpha, rho, nu = map(float, res.x)
        b_hat = float(beta)
    rmse = float(np.sqrt(np.mean(res.fun**2)))
    return {
        "alpha": alpha,
        "beta": b_hat,
        "rho": rho,
        "nu": nu,
        "rmse": rmse,
        "n_strikes": float(k.size),
        "converged": float(res.success),
    }


def sabr_rho_nu_from_smile(
    forward: float,
    tenor: float,
    strikes: Array,
    market_vols: Array,
    beta: float = 0.5,
) -> dict[str, float]:
    """Two-step smile decomposition: fix beta, fit (rho, nu) to the smile
    shape with alpha re-solved from ATM each step (Hagan's ATM trick)."""
    k = np.asarray(strikes, dtype=float).ravel()
    v = np.asarray(market_vols, dtype=float).ravel()
    if k.shape != v.shape or k.size < 3:
        raise ValueError("strikes and market_vols must match, >= 3")
    atm_idx = int(np.argmin(np.abs(k - forward)))
    sigma_atm = float(v[atm_idx])

    def resid(theta: Array) -> Array:
        rho, nu = float(theta[0]), float(theta[1])
        try:
            alpha = sabr_alpha_from_atm(forward, tenor, sigma_atm, beta, rho, nu)
        except ValueError:
            return np.full_like(v, 1e3)
        return np.asarray(
            sabr_implied_vol(forward, k, tenor, alpha, beta, rho, nu) - v, dtype=float
        )

    res = optimize.least_squares(
        resid, np.array([0.0, 0.5]), bounds=([-0.999, 1e-8], [0.999, np.inf])
    )
    rho, nu = float(res.x[0]), float(res.x[1])
    alpha = sabr_alpha_from_atm(forward, tenor, sigma_atm, beta, rho, nu)
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "rho": rho,
        "nu": nu,
        "rmse": float(np.sqrt(np.mean(res.fun**2))),
    }
