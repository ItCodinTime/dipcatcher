"""Jump-diffusion models: Merton (1976) option series + path simulation,
Kou (2002) double-exponential jumps.

Merton JD: dS/S = (r - lam*kappa) dt + sigma dW + (J - 1) dN, with
ln J ~ N(mu_j, s_j^2) and kappa = E[J-1]. The call price is a Poisson-
weighted BSM series (exact, converges quickly for lam*T ~ O(10)).

Kou: ln J = +V^+ w.p. p (V^+ ~ Exp(eta1)), -V^- w.p. 1-p
(V^- ~ Exp(eta2)); closed-form pricing needs the Hh functions, so this
module provides simulation + analytic moments only.

Fail-closed on non-finite inputs, non-positive S/K/T/sigma/lam/s_j.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, poisson

Array = NDArray[np.float64]


def _bsm_call(s: float, k: float, t: float, r: float, sigma: float) -> float:
    if t <= 0.0 or sigma <= 0.0:
        return max(s - k * math.exp(-r * t), 0.0)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return float(s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2))


def _check_jd(s: float, k: float, t: float, sigma: float, lam: float, s_j: float) -> None:
    vals = [s, k, t, sigma, lam, s_j]
    if not np.isfinite(vals).all():
        raise ValueError("inputs must be finite")
    if s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0 or s_j <= 0.0 or lam < 0.0:
        raise ValueError("S,K,T,sigma,s_j > 0 and lam >= 0 required")


def merton_kappa(mu_j: float, s_j: float) -> float:
    """E[J-1] = exp(mu_j + s_j^2 / 2) - 1 for lognormal jumps."""
    if not np.isfinite([mu_j, s_j]).all() or s_j < 0.0:
        raise ValueError("mu_j finite, s_j >= 0")
    return float(math.exp(mu_j + 0.5 * s_j * s_j) - 1.0)


def merton_jump_call(
    spot: float,
    strike: float,
    tenor: float,
    rate: float,
    sigma: float,
    lam: float,
    mu_j: float,
    s_j: float,
    n_max: int | None = None,
    tol: float = 1e-12,
) -> float:
    """Merton (1976) call price via the Poisson-weighted BSM series.

    For each n jumps: sigma_n^2 = sigma^2 + n s_j^2 / T and shifted
    rate r_n = r - lam*kappa + n (mu_j + s_j^2/2) / T.
    """
    _check_jd(spot, strike, tenor, sigma, lam, s_j)
    if not np.isfinite([rate, mu_j]).all():
        raise ValueError("rate, mu_j must be finite")
    kappa = merton_kappa(mu_j, s_j)
    lam_t = lam * tenor
    if lam_t == 0.0:
        return _bsm_call(spot, strike, tenor, rate, sigma)
    if n_max is None:
        n_max = int(max(40, poisson.isf(tol, lam_t)))
    if n_max < 1:
        raise ValueError("n_max >= 1")
    total = 0.0
    for n in range(n_max + 1):
        w = float(poisson.pmf(n, lam_t))
        if w < tol and n > lam_t:
            break
        sig_n = math.sqrt(sigma * sigma + n * s_j * s_j / tenor)
        r_n = rate - lam * kappa + n * (mu_j + 0.5 * s_j * s_j) / tenor
        total += w * _bsm_call(spot, strike, tenor, r_n, sig_n)
    return float(total)


def merton_jump_put(
    spot: float,
    strike: float,
    tenor: float,
    rate: float,
    sigma: float,
    lam: float,
    mu_j: float,
    s_j: float,
) -> float:
    """Put via put-call parity: P = C - S + K e^{-rT} (model-consistent)."""
    call = merton_jump_call(spot, strike, tenor, rate, sigma, lam, mu_j, s_j)
    return float(call - spot + strike * math.exp(-rate * tenor))


def merton_jump_simulate(
    spot: float,
    tenor: float,
    rate: float,
    sigma: float,
    lam: float,
    mu_j: float,
    s_j: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Array:
    """Exact simulation on a grid: log-Euler diffusion + compound jumps.

    Returns (n_paths, n_steps+1) paths including S_0.
    """
    _check_jd(spot, 1.0, tenor, sigma, lam, s_j)
    if not np.isfinite([rate, mu_j]).all():
        raise ValueError("rate, mu_j must be finite")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps, n_paths >= 1")
    kappa = merton_kappa(mu_j, s_j)
    dt = tenor / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = spot
    log_s = np.log(paths[:, 0])
    drift = (rate - lam * kappa - 0.5 * sigma * sigma) * dt
    for i in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        nj = rng.poisson(lam * dt, size=n_paths)
        jump = np.zeros(n_paths)
        active = nj > 0
        if active.any():
            # sum of nj iid N(mu_j, s_j^2) = N(nj mu_j, nj s_j^2)
            jump[active] = rng.normal(nj[active] * mu_j, np.sqrt(nj[active]) * s_j)
        log_s = log_s + drift + sigma * math.sqrt(dt) * z + jump
        paths[:, i] = np.exp(log_s)
    return paths


def merton_log_moments(
    tenor: float,
    rate: float,
    sigma: float,
    lam: float,
    mu_j: float,
    s_j: float,
) -> dict[str, float]:
    """Mean/variance/skew of ln(S_T/S_0) under Merton JD."""
    _check_jd(1.0, 1.0, tenor, sigma, lam, s_j)
    if not np.isfinite([rate, mu_j]).all():
        raise ValueError("rate, mu_j must be finite")
    kappa = merton_kappa(mu_j, s_j)
    lam_t = lam * tenor
    mean = (rate - lam * kappa - 0.5 * sigma * sigma) * tenor + lam_t * mu_j
    var = sigma * sigma * tenor + lam_t * (mu_j * mu_j + s_j * s_j)
    # third central moment of compound-Poisson + Gaussian
    m3 = lam_t * (mu_j**3 + 3.0 * mu_j * s_j * s_j)
    skew = m3 / var**1.5 if var > 0 else float("nan")
    return {"mean": float(mean), "var": float(var), "skew": float(skew)}


def kou_simulate(
    spot: float,
    tenor: float,
    rate: float,
    sigma: float,
    lam: float,
    p_up: float,
    eta1: float,
    eta2: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Array:
    """Kou (2002) double-exponential jump-diffusion paths.

    ln J = +Exp(eta1) w.p. p_up, -Exp(eta2) w.p. 1-p_up; kappa adjusted
    so the discounted price is a martingale.
    """
    vals = [spot, tenor, rate, sigma, lam, p_up, eta1, eta2]
    if not np.isfinite(vals).all():
        raise ValueError("inputs must be finite")
    if spot <= 0.0 or tenor <= 0.0 or sigma <= 0.0 or lam < 0.0:
        raise ValueError("spot, tenor, sigma > 0, lam >= 0")
    if not 0.0 < p_up < 1.0 or eta1 <= 1.0 or eta2 <= 0.0:
        raise ValueError("p_up in (0,1), eta1 > 1 (finite mean), eta2 > 0")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps, n_paths >= 1")
    kappa = p_up * eta1 / (eta1 - 1.0) + (1.0 - p_up) * eta2 / (eta2 + 1.0) - 1.0
    dt = tenor / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = spot
    log_s = np.log(paths[:, 0])
    drift = (rate - lam * kappa - 0.5 * sigma * sigma) * dt
    for i in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        nj = rng.poisson(lam * dt, size=n_paths)
        active = nj > 0
        jump = np.zeros(n_paths)
        if active.any():
            n_jumps = int(nj.sum())
            signs = rng.random(n_jumps) < p_up
            mags = np.where(
                signs,
                rng.exponential(1.0 / eta1, size=n_jumps),
                -rng.exponential(1.0 / eta2, size=n_jumps),
            )
            idx = np.repeat(np.nonzero(active)[0], nj[active])
            np.add.at(jump, idx, mags)
        log_s = log_s + drift + sigma * math.sqrt(dt) * z + jump
        paths[:, i] = np.exp(log_s)
    return paths


def kou_kappa(p_up: float, eta1: float, eta2: float) -> float:
    """E[J-1] under double-exponential jumps."""
    if not np.isfinite([p_up, eta1, eta2]).all():
        raise ValueError("inputs must be finite")
    if not 0.0 <= p_up <= 1.0 or eta1 <= 1.0 or eta2 <= 0.0:
        raise ValueError("p_up in [0,1], eta1 > 1, eta2 > 0")
    return float(p_up * eta1 / (eta1 - 1.0) + (1.0 - p_up) * eta2 / (eta2 + 1.0) - 1.0)
