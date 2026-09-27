"""Alpha-stable (Levy stable) distribution: simulation, ECF fit, density.

Stable laws are the only possible limits of normalised sums of i.i.d. variables
(generalised CLT) and model heavy tails and skew with four parameters:
stability ``alpha in (0, 2]``, skewness ``beta in [-1, 1]``, scale ``c > 0`` and
location ``loc`` (``alpha = 2`` is Gaussian).

- ``stable_rvs`` uses the Chambers-Mallows-Stuck (1976) generator.
- ``stable_fit_ecf`` estimates ``alpha`` and ``c`` for (near-)symmetric data
  from the empirical characteristic function via the Press (1972) /
  Koutrouvelis (1980) log-log regression ``log(-log|phi(t)|) = alpha log|t| +
  alpha log c``; ``loc`` is the sample median.
- ``stable_pdf`` / ``stable_cdf`` wrap SciPy's ``levy_stable`` (S1
  parameterisation).

References: J. M. Chambers, C. L. Mallows, B. W. Stuck (1976), JASA;
S. J. Press (1972), JASA; A. Koutrouvelis (1980), JASA.  Fail-closed on invalid
parameters or non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import levy_stable

Array = NDArray[np.float64]


def _check_params(alpha: float, beta: float, c: float) -> None:
    if not 0.0 < alpha <= 2.0:
        raise ValueError("alpha must be in (0, 2]")
    if not -1.0 <= beta <= 1.0:
        raise ValueError("beta must be in [-1, 1]")
    if c <= 0.0:
        raise ValueError("scale c must be positive")


def stable_rvs(
    alpha: float,
    beta: float = 0.0,
    c: float = 1.0,
    loc: float = 0.0,
    size: int = 1000,
    rng: np.random.Generator | None = None,
) -> Array:
    """Chambers-Mallows-Stuck sampler for S(alpha, beta, c, loc)."""
    _check_params(alpha, beta, c)
    gen = np.random.default_rng() if rng is None else rng
    u = (gen.random(size) - 0.5) * np.pi  # Uniform(-pi/2, pi/2)
    w = gen.exponential(1.0, size)
    if abs(alpha - 1.0) < 1e-9:
        x = (2.0 / np.pi) * (
            (np.pi / 2.0 + beta * u) * np.tan(u)
            - beta * np.log((np.pi / 2.0 * w * np.cos(u)) / (np.pi / 2.0 + beta * u))
        )
    else:
        zeta = -beta * np.tan(np.pi * alpha / 2.0)
        xi = np.arctan(-zeta) / alpha
        term1 = np.sin(alpha * (u + xi)) / np.cos(u) ** (1.0 / alpha)
        term2 = (np.cos(u - alpha * (u + xi)) / w) ** ((1.0 - alpha) / alpha)
        x = (1.0 + zeta**2) ** (1.0 / (2.0 * alpha)) * term1 * term2
    return np.asarray(c * x + loc, dtype=float)


def stable_pdf(x: Array, alpha: float, beta: float, loc: float = 0.0, scale: float = 1.0) -> Array:
    """Stable density via SciPy ``levy_stable`` (S1 parameterisation)."""
    _check_params(alpha, beta, scale)
    return np.asarray(levy_stable.pdf(np.asarray(x, dtype=float), alpha, beta, loc, scale), float)


def stable_cdf(x: Array, alpha: float, beta: float, loc: float = 0.0, scale: float = 1.0) -> Array:
    """Stable CDF via SciPy ``levy_stable``."""
    _check_params(alpha, beta, scale)
    return np.asarray(levy_stable.cdf(np.asarray(x, dtype=float), alpha, beta, loc, scale), float)


def stable_fit_ecf(x: Array, n_points: int = 12) -> dict[str, float]:
    """Estimate ``alpha``, ``c``, ``loc`` for symmetric data via the ECF regression."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 50 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 50 observations")
    loc = float(np.median(arr))
    y = arr - loc
    scale0 = float(np.median(np.abs(y))) + 1e-12
    t = np.linspace(0.1, 1.0, n_points) / scale0
    re = np.array([np.mean(np.cos(tk * y)) for tk in t])
    im = np.array([np.mean(np.sin(tk * y)) for tk in t])
    phi_abs = np.sqrt(re**2 + im**2)
    ok = (phi_abs > 0.03) & (phi_abs < 0.999)
    if ok.sum() < 3:
        raise ValueError("insufficient usable ECF points to fit")
    lt = np.log(t[ok])
    w = np.log(-np.log(phi_abs[ok]))
    design = np.column_stack([np.ones(lt.size), lt])
    coef, *_ = np.linalg.lstsq(design, w, rcond=None)
    intercept, alpha = float(coef[0]), float(coef[1])
    alpha = float(np.clip(alpha, 0.1, 2.0))
    c = float(np.exp(intercept / alpha)) if alpha > 0 else float("nan")
    return {"alpha": alpha, "c": c, "loc": loc}
