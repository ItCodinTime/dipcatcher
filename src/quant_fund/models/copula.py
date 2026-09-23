"""Copula models for dependence structure (bivariate focus).

References:
- Sklar (1959). Fonctions de repartition a n dimensions — C(u,v) separates
  marginals from dependence.
- Nelsen (2006). *An Introduction to Copulas* — Archimedean generators,
  Kendall tau inversions, tail-dependence coefficients.
- Genest, Favre (2007). Everything you always wanted to know about copula
  modeling — pseudo-likelihood (IFM) estimation.
- Embrechts, McNeil, Straumann (2002). Correlation and dependence in risk
  management — Gaussian/t copulas and tail dependence.
- Demarta, McNeil (2005). The t copula and related copulas.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt
from scipy import stats as sstats

Array = NDArray[np.float64]


def _as_pairs(u: Array, name: str = "u") -> Array:
    m = np.asarray(u, dtype=float)
    if m.ndim != 2 or m.shape[1] != 2 or not np.all(np.isfinite(m)):
        raise ValueError(f"{name} must be a finite n x 2 array")
    if m.shape[0] < 10:
        raise ValueError(f"{name} must have >= 10 observations")
    if np.any(m < 0.0) or np.any(m > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return m


def pseudo_observations(x: Array) -> Array:
    """Rank-transform n x d data to (0,1) pseudo-observations: ``rank/(n+1)``."""
    m = np.asarray(x, dtype=float)
    if m.ndim != 2 or not np.all(np.isfinite(m)):
        raise ValueError("x must be a finite n x d array")
    if m.shape[0] < 10 or m.shape[1] < 2:
        raise ValueError("x must have >= 10 rows and >= 2 columns")
    return sstats.rankdata(m, axis=0) / (m.shape[0] + 1.0)


def kendall_tau(u: Array) -> float:
    """Kendall's tau of a bivariate (pseudo-)sample."""
    m = _as_pairs(u)
    tau = float(sstats.kendalltau(m[:, 0], m[:, 1]).statistic)
    if not np.isfinite(tau):
        raise ValueError("Kendall tau is undefined (constant input)")
    return tau


def empirical_tail_dependence(u: Array, k: int | None = None) -> tuple[float, float]:
    """Empirical lower/upper tail-dependence estimates.

    ``lambda_L = P(v <= p | u <= p)``, ``lambda_U = P(v > 1-p | u > 1-p)``
    with ``p = k/n`` (default ``k = sqrt(n)``).
    """
    m = _as_pairs(u)
    n = m.shape[0]
    kk = int(max(2, math.sqrt(n))) if k is None else int(k)
    if kk < 1 or kk >= n // 2:
        raise ValueError("k must be in [1, n/2)")
    s = np.sort(m[:, 0])
    p = s[kk]
    q = s[n - 1 - kk]
    lower = float(np.mean(m[:, 1][m[:, 0] <= p] <= p))
    upper = float(np.mean(m[:, 1][m[:, 0] >= q] >= q))
    return lower, upper


def fit_gaussian_copula(u: Array) -> float:
    """IFM correlation for the Gaussian copula: ``sin(pi/2 * tau)`` check via
    direct normal-score correlation (more accurate than tau inversion).
    """
    m = _as_pairs(u)
    z = sstats.norm.ppf(np.clip(m, 1e-9, 1.0 - 1e-9))
    rho = float(np.corrcoef(z[:, 0], z[:, 1])[0, 1])
    if not np.isfinite(rho):
        raise ValueError("Gaussian copula fit failed")
    return float(np.clip(rho, -0.999, 0.999))


def fit_t_copula(u: Array, nu_grid: Array | None = None) -> tuple[float, float]:
    """IFM for the t copula: MLE over (rho, nu) via t-score log-likelihood.

    Returns ``(rho, nu)``.  ``nu_grid`` optionally restricts nu to a grid
    (2.1–30 by default) to stabilize the flat likelihood tail.
    """
    m = _as_pairs(u)
    nus = (
        np.geomspace(2.1, 30.0, 25)
        if nu_grid is None
        else np.asarray(nu_grid, dtype=float).reshape(-1)
    )
    if nus.size < 2 or not np.all(np.isfinite(nus)) or np.any(nus <= 2.0):
        raise ValueError("nu_grid must be a finite vector with values > 2")
    from scipy.special import gammaln

    def _make_nll(z: Array, nu: float, const: float, marg: float):
        n_obs = z.shape[0]

        def _nll(r: float) -> float:
            rho = float(np.clip(r, -0.98, 0.98))
            d = (z[:, 0] ** 2 + z[:, 1] ** 2 - 2 * rho * z[:, 0] * z[:, 1]) / (1.0 - rho**2)
            return -(
                const
                + marg
                - 0.5 * n_obs * math.log(1.0 - rho**2)
                - (nu + 2.0) / 2.0 * float(np.sum(np.log1p(d / nu)))
            )

        return _nll

    best = (-np.inf, 0.0, float(nus[0]))
    for nu in nus:
        z = sstats.t.ppf(np.clip(m, 1e-9, 1.0 - 1e-9), df=nu)
        if not np.all(np.isfinite(z)):
            continue
        n_obs = z.shape[0]
        # Constant part of the t-copula log-density (does not depend on rho).
        const = n_obs * (
            gammaln((nu + 2.0) / 2.0)
            - gammaln(nu / 2.0)
            - math.log(nu * math.pi)
            - 2.0 * (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * math.log(nu * math.pi))
        )
        marg = -((nu + 1.0) / 2.0) * float(np.sum(np.log1p(z**2 / nu)))
        nll = _make_nll(z, float(nu), const, marg)
        res = opt.minimize_scalar(nll, bounds=(-0.98, 0.98), method="bounded")
        if -res.fun > best[0]:
            best = (-res.fun, float(res.x), float(nu))
    if not np.isfinite(best[1]):
        raise ValueError("t copula fit failed")
    return best[1], best[2]


def fit_clayton(u: Array) -> float:
    """Clayton theta via Kendall inversion: ``theta = 2 tau / (1 - tau)``."""
    tau = kendall_tau(u)
    if tau >= 0.999 or tau <= -0.999:
        raise ValueError("tau near ±1 — Clayton theta undefined")
    theta = 2.0 * tau / (1.0 - tau)
    if theta <= -1.0:
        raise ValueError("inverted theta <= -1 — inconsistent with Clayton")
    return theta


def fit_gumbel(u: Array) -> float:
    """Gumbel alpha via Kendall inversion: ``alpha = 1 / (1 - tau)``."""
    tau = kendall_tau(u)
    if tau >= 0.999:
        raise ValueError("tau near 1 — Gumbel alpha diverges")
    alpha = 1.0 / (1.0 - tau)
    if alpha < 1.0:
        raise ValueError("inverted alpha < 1 — inconsistent with Gumbel")
    return alpha


def gaussian_copula_sim(rho: float, n: int, seed: int | None = None) -> Array:
    """Simulate n uniform pairs from the Gaussian copula."""
    r = _finite_rho(rho)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, r], [r, 1.0]])
    z = rng.multivariate_normal(np.zeros(2), cov, size=n)
    return sstats.norm.cdf(z)


def t_copula_sim(rho: float, nu: float, n: int, seed: int | None = None) -> Array:
    """Simulate n uniform pairs from the t copula."""
    r = _finite_rho(rho)
    if not np.isfinite(nu) or nu <= 2.0:
        raise ValueError("nu must be > 2")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, r], [r, 1.0]])
    z = rng.multivariate_normal(np.zeros(2), cov, size=n)
    w = rng.chisquare(nu, size=n) / nu
    t = z / np.sqrt(w)[:, None]
    return sstats.t.cdf(t, df=nu)


def clayton_copula_sim(theta: float, n: int, seed: int | None = None) -> Array:
    """Marshall–Olkin simulation of the Clayton copula (theta > 0)."""
    th = _finite_scalar(theta, "theta")
    if th <= 0.0:
        raise ValueError("theta must be positive (lower-tail Clayton)")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    e1 = rng.exponential(1.0, n)
    e2 = rng.exponential(1.0, n)
    g = rng.gamma(1.0 / th, 1.0, n)  # Laplace-transform variable
    u = (1.0 + e1 / g) ** (-1.0 / th)
    v = (1.0 + e2 / g) ** (-1.0 / th)
    return np.column_stack([u, v])


def gumbel_copula_sim(alpha: float, n: int, seed: int | None = None) -> Array:
    """Gumbel simulation via the stable-frailty Marshall–Olkin construction.

    Uses the Laplace transform of a ``1/alpha``-stable variable sampled via
    the Chambers–Mallows–Stuck method.
    """
    a = _finite_scalar(alpha, "alpha")
    if a < 1.0:
        raise ValueError("alpha must be >= 1 for Gumbel")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if a == 1.0:
        rng = np.random.default_rng(seed)
        return rng.random((n, 2))
    rng = np.random.default_rng(seed)
    # Frailty V ~ Stable(1/alpha, 1, gamma=cos^{alpha}(pi/(2 alpha)), 0) via
    # the Chambers–Mallows–Stuck construction (Nolan param. "0").
    st = 1.0 / a
    u0 = rng.uniform(-math.pi / 2.0, math.pi / 2.0, n)
    w = rng.exponential(1.0, n)
    theta = u0 + math.pi / 2.0
    part1 = np.sin(st * theta) / np.cos(u0) ** (1.0 / st)
    part2 = (np.cos(u0 - st * theta) / w) ** ((1.0 - st) / st)
    stable_unit = part1 * part2
    gamma_scale = math.cos(math.pi * st / 2.0) ** a
    frailty = np.maximum(stable_unit * gamma_scale ** (1.0 / st), 1e-12)
    e1 = rng.exponential(1.0, n)
    e2 = rng.exponential(1.0, n)
    u = np.exp(-((e1 / frailty) ** (1.0 / a)))
    v = np.exp(-((e2 / frailty) ** (1.0 / a)))
    return np.column_stack([np.clip(u, 1e-12, 1.0), np.clip(v, 1e-12, 1.0)])


def _finite_rho(rho: float) -> float:
    r = float(rho)
    if not np.isfinite(r) or not (-1.0 < r < 1.0):
        raise ValueError("rho must be finite and in (-1, 1)")
    return r


def _finite_scalar(x: float, name: str) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def copula_tail_dependence_theory(
    family: str, param: float, nu: float | None = None
) -> tuple[float, float]:
    """Analytic lower/upper tail dependence for the fitted family.

    Gaussian: (0, 0).  t: ``2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))``.
    Clayton: ``(2^{-1/theta}, 0)``.  Gumbel: ``(0, 2 - 2^{1/alpha})``.
    """
    fam = family.lower()
    if fam == "gaussian":
        _finite_rho(param)
        return 0.0, 0.0
    if fam == "t":
        r = _finite_rho(param)
        if nu is None or not np.isfinite(nu) or nu <= 0.0:
            raise ValueError("t copula requires finite nu > 0")
        lam = 2.0 * float(sstats.t.cdf(-math.sqrt((nu + 1.0) * (1.0 - r) / (1.0 + r)), df=nu + 1.0))
        return lam, lam
    if fam == "clayton":
        th = _finite_scalar(param, "theta")
        if th <= 0.0:
            raise ValueError("theta must be positive")
        return float(2.0 ** (-1.0 / th)), 0.0
    if fam == "gumbel":
        a = _finite_scalar(param, "alpha")
        if a < 1.0:
            raise ValueError("alpha must be >= 1")
        return 0.0, float(2.0 - 2.0 ** (1.0 / a))
    raise ValueError(f"unknown copula family: {family!r}")
