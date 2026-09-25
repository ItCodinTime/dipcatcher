"""One-factor Gaussian-copula portfolio default distribution.

Li (2000) modelled joint defaults with a Gaussian copula.  In the one-factor
version each name's latent variable is ``sqrt(rho) M + sqrt(1-rho) Z_i`` and it
defaults when this falls below ``c_i = Phi^{-1}(pd_i)``.  Conditional on the
common factor ``M = m`` defaults are independent with probability

    p_i(m) = Phi( (c_i - sqrt(rho) m) / sqrt(1 - rho) ),

and the conditional loss distribution is built by the Andersen-Sidenius-Basu
(2003) convolution recursion.  Integrating over ``M`` with Gauss-Hermite
quadrature gives the unconditional distribution of the number of defaults.

References: D. Li (2000), Journal of Fixed Income; L. Andersen, J. Sidenius,
S. Basu (2003), Risk.  Fail-closed on invalid parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]


def conditional_default_prob(pd: Array, rho: float, m: float) -> Array:
    """Factor-conditional default probabilities ``p_i(m)``."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be in (0, 1)")
    c = norm.ppf(np.asarray(pd, dtype=float))
    return norm.cdf((c - np.sqrt(rho) * m) / np.sqrt(1.0 - rho))


def _conditional_pmf(pvec: Array) -> Array:
    """Distribution of the number of defaults given independent probs (ASB recursion)."""
    pmf = np.array([1.0])
    for p in pvec:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


def portfolio_default_distribution(
    pd: Array, rho: float, n_nodes: int = 64
) -> dict[str, Array | float]:
    """Unconditional pmf of the number of defaults over a heterogeneous pool."""
    pdv = np.asarray(pd, dtype=float).ravel()
    if pdv.size < 1 or not np.isfinite(pdv).all() or ((pdv <= 0) | (pdv >= 1)).any():
        raise ValueError("pd must be finite probabilities in (0, 1)")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be in (0, 1)")
    nodes, weights = hermegauss(n_nodes)
    norm_w = weights / np.sqrt(2.0 * np.pi)
    n = pdv.size
    pmf = np.zeros(n + 1)
    for m, w in zip(nodes, norm_w, strict=True):
        pmf += w * _conditional_pmf(conditional_default_prob(pdv, rho, float(m)))
    pmf = np.clip(pmf, 0.0, None)
    pmf /= pmf.sum()
    k = np.arange(n + 1, dtype=float)
    mean = float(np.sum(k * pmf))
    var = float(np.sum((k - mean) ** 2 * pmf))
    return {"pmf": pmf, "mean": mean, "var": var}
