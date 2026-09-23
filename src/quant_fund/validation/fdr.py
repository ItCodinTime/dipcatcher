"""False-discovery-rate and family-wise multiple-testing corrections.

Companion to :mod:`quant_fund.metrics.snooping` (White RC / SPA / StepM / MCS):
those control via resampling; these control analytically from the p-value
vector alone.

References:
- Bonferroni (1936); Šidák (1967) — single-step FWER.
- Holm (1979). A simple sequentially rejective multiple test procedure.
  *Scand. J. Statist.* 6 — step-down FWER.
- Hochberg (1988). A sharper Bonferroni procedure. *Biometrika* 75 — step-up FWER.
- Simes (1986). An improved Bonferroni procedure for multiple tests.
- Benjamini, Hochberg (1995). Controlling the false discovery rate. *JRSS-B* 57.
- Benjamini, Yekutieli (2001). The control of the FDR under dependency.
  *Ann. Statist.* 29 — arbitrary dependence via the c(m) harmonic factor.
- Storey (2002). A direct approach to false discovery rates. *JRSS-B* 64 —
  pi0 estimate and q-values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
AdjMap = dict[str, Array | BoolArray]

_METHODS = ("bonferroni", "sidak", "holm", "hochberg", "simes", "bh", "by")


def _pvals(p: Array, name: str = "pvals") -> Array:
    v = np.asarray(p, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(v).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(v < 0.0) or np.any(v > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return v


def _order_restore(order: NDArray[np.intp], values: Array) -> Array:
    out = np.empty_like(values)
    out[order] = values
    return out


def bonferroni(pvals: Array, alpha: float = 0.05) -> AdjMap:
    """Bonferroni single-step FWER: ``adj_i = min(1, m * p_i)``."""
    p = _pvals(pvals)
    m = p.size
    adj = np.minimum(1.0, m * p)
    return {"adjusted": adj, "rejected": adj <= alpha}


def sidak(pvals: Array, alpha: float = 0.05) -> AdjMap:
    """Šidák (1967) single-step: ``adj_i = 1 - (1 - p_i)^m``."""
    p = _pvals(pvals)
    adj = 1.0 - (1.0 - p) ** p.size
    return {"adjusted": adj, "rejected": adj <= alpha}


def holm(pvals: Array, alpha: float = 0.05) -> AdjMap:
    """Holm (1979) step-down FWER (uniformly more powerful than Bonferroni)."""
    p = _pvals(pvals)
    m = p.size
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adj = _order_restore(order, np.minimum(1.0, adj_sorted))
    return {"adjusted": adj, "rejected": adj <= alpha}


def hochberg(pvals: Array, alpha: float = 0.05) -> AdjMap:
    """Hochberg (1988) step-up FWER — dominates Holm under independence."""
    p = _pvals(pvals)
    m = p.size
    order = np.argsort(p)[::-1]  # descending
    adj_desc = np.minimum.accumulate((np.arange(1, m + 1)) * p[order])
    adj = _order_restore(order, np.minimum(1.0, adj_desc))
    return {"adjusted": adj, "rejected": adj <= alpha}


def simes(pvals: Array) -> float:
    """Simes (1986) global-null statistic ``min_i m * p_(i) / i``."""
    p = _pvals(pvals)
    m = p.size
    srt = np.sort(p)
    return float(np.min(m * srt / np.arange(1, m + 1)))


def benjamini_hochberg(pvals: Array, alpha: float = 0.05) -> AdjMap:
    """Benjamini–Hochberg (1995) FDR control, valid under PRDS dependence."""
    p = _pvals(pvals)
    m = p.size
    order = np.argsort(p)
    raw = m * p[order] / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adj = _order_restore(order, np.minimum(1.0, adj_sorted))
    return {"adjusted": adj, "rejected": adj <= alpha}


def benjamini_yekutieli(pvals: Array, alpha: float = 0.05) -> AdjMap:
    """Benjamini–Yekutieli (2001) FDR under arbitrary dependence (c(m) factor)."""
    p = _pvals(pvals)
    m = p.size
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p)
    raw = m * c_m * p[order] / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adj = _order_restore(order, np.minimum(1.0, adj_sorted))
    return {"adjusted": adj, "rejected": adj <= alpha}


def storey_pi0(pvals: Array, lambdas: Array | None = None) -> float:
    """Storey (2002) null-proportion estimate ``pi0 = #(p > lam) / (m (1-lam))``.

    Uses the smoothed minimum over the lambda grid (Storey's recommended
    estimator), clipped to [0, 1].
    """
    p = _pvals(pvals)
    grid = np.asarray(lambdas, dtype=float) if lambdas is not None else np.arange(0.05, 0.95, 0.05)
    grid = grid[np.isfinite(grid) & (grid >= 0.0) & (grid < 1.0)]
    if grid.size == 0:
        raise ValueError("lambdas must contain values in [0, 1)")
    pi0s = np.array([np.mean(p > lam) / (1.0 - lam) for lam in grid])
    return float(min(1.0, np.min(pi0s)))


def storey_qvalues(pvals: Array, lambdas: Array | None = None) -> dict[str, Array]:
    """Storey (2002) q-values ``q_(i) = pi0 * m * p_(i) / i`` (monotone-adjusted)."""
    p = _pvals(pvals)
    m = p.size
    pi0 = storey_pi0(p, lambdas)
    order = np.argsort(p)
    raw = pi0 * m * p[order] / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adj = _order_restore(order, np.minimum(1.0, adj_sorted))
    return {"qvalues": adj, "pi0": np.full(m, pi0)}


def adjust(pvals: Array, method: str = "bh", alpha: float = 0.05) -> AdjMap:
    """Dispatcher over the correction family.

    ``method`` in ``{bonferroni, sidak, holm, hochberg, simes, bh, by, storey}``;
    ``simes`` returns the global statistic only (no rejections).
    """
    if isinstance(method, bool) or not isinstance(method, str):
        raise ValueError("method must be a string")
    key = method.lower()
    if key == "simes":
        return {"adjusted": np.array([simes(pvals)]), "rejected": np.array([False])}
    if key == "storey":
        q = storey_qvalues(pvals)
        return q | {"rejected": q["qvalues"] <= alpha}
    if key == "bh":
        return benjamini_hochberg(pvals, alpha)
    if key == "by":
        return benjamini_yekutieli(pvals, alpha)
    if key == "holm":
        return holm(pvals, alpha)
    if key == "hochberg":
        return hochberg(pvals, alpha)
    if key == "bonferroni":
        return bonferroni(pvals, alpha)
    if key == "sidak":
        return sidak(pvals, alpha)
    raise ValueError(f"method must be one of {_METHODS + ('storey',)}")
