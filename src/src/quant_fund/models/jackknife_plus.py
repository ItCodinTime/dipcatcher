"""Jackknife+ leave-one-out conformal wrapper.

Barber, Candès, Ramdas, Tibshirani (2021). Finite-sample coverage is ≥ 1-2α.
That floor is **marginal under exchangeability**, not training-conditional
(Bian & Barber 2023 caveat). Does not reimplement QR. Wraps a residual /
quantile predictor, or an exact LOO mean±z Gaussian band that updates in O(n).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.models.base import JoblibMixin, ModelMeta

Array = NDArray[np.float64]
ResidualModel = Callable[[Array], Array]


def _as_1d(values: Array) -> Array:
    return np.asarray(values, dtype=float).reshape(-1)


def _align(y: Array, other: Array, name: str) -> tuple[Array, Array]:
    y = _as_1d(y)
    extra = np.asarray(other, dtype=float).reshape(-1)
    if extra.size == 1 and y.size != 1:
        extra = np.full(y.size, float(extra.item()))
    if y.size != extra.size:
        raise ValueError(f"y and {name} must have the same length")
    mask = np.isfinite(y) & np.isfinite(extra)
    return y[mask], extra[mask]


def loo_mean_and_scale(y: Array) -> tuple[Array, Array]:
    """Closed-form leave-one-out mean and unbiased std of a 1-d sample.

    μ_{-i} = (n μ - y_i)/(n-1)
    (n-2) σ_{-i}^2 = Σ_j (y_j-μ)^2 - n/(n-1) (y_i-μ)^2
    For n=2 the LOO std is undefined; scale is returned as 0.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.size)
    if n < 2:
        # Leave-one-out mean of a single observation is undefined (0/0);
        # fail closed rather than emit NaN + a RuntimeWarning.
        return np.full(n, np.nan), np.zeros(n, dtype=float)
    mu = float(np.mean(y))
    loc = (n * mu - y) / (n - 1)
    if n < 3:
        return loc, np.zeros(n, dtype=float)
    ssd = float(np.sum((y - mu) ** 2))
    loo_ssd = ssd - (n / (n - 1)) * (y - mu) ** 2
    scale = np.sqrt(np.maximum(loo_ssd / (n - 2), 0.0))
    return loc, scale


def _jackknife_plus_quantiles(lo_ens: Array, hi_ens: Array, alpha: float) -> tuple[Array, Array]:
    """Paper (9): q^- of {μ_{-i}-R_i} and q^+ of {μ_{-i}+R_i}."""
    m = int(lo_ens.shape[0])
    out_lo = np.empty(m, dtype=float)
    out_hi = np.empty(m, dtype=float)
    for i in range(m):
        row_lo = lo_ens[i]
        row_hi = hi_ens[i]
        finite = np.isfinite(row_lo) & np.isfinite(row_hi)
        if not bool(np.any(finite)):
            out_lo[i] = np.nan
            out_hi[i] = np.nan
            continue
        out_lo[i] = -conformal_quantile(-row_lo[finite], alpha)
        out_hi[i] = conformal_quantile(row_hi[finite], alpha)
    return out_lo, out_hi


class JackknifePlus(JoblibMixin):
    """Leave-one-out conformal wrapper. Guarantee is 1-2α, not 1-α.

    Floor is marginal under exchangeability (Barber–Candès 2021),
    not training-conditional (Bian–Barber 2023).
    """

    def __init__(
        self,
        alpha: float = 0.10,
        residual_model: ResidualModel | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.residual_model = residual_model
        self.z = float(norm.ppf(1.0 - self.alpha / 2.0))
        self.loo_loc_: Array | None = None
        self.loo_scale_: Array | None = None
        self.scores_: Array | None = None
        self.n_ = 0

    def _store_loo(self, residual: Array) -> JackknifePlus:
        n = int(residual.size)
        if n == 0:
            raise ValueError("Jackknife+ needs a non-empty sample")
        if n < 2:
            raise ValueError("Jackknife+ needs n>=2 for leave-one-out")
        loc, scale = loo_mean_and_scale(residual)
        self.loo_loc_ = loc
        self.loo_scale_ = scale
        # Classic R_i = |Y_i - μ_{-i}|. Not CQR vs an already-wide z-band.
        self.scores_ = np.abs(residual - loc)
        self.n_ = n
        return self

    def fit(self, y: Array, pred: Array | None = None) -> JackknifePlus:
        """Exact LOO mean±z band on residuals y - pred (pred may be a point forecast)."""
        y = _as_1d(y)
        if pred is None and self.residual_model is not None:
            pred = self.residual_model(y)
        if pred is None:
            pred = np.zeros_like(y)
        y, pred = _align(y, pred, "pred")
        return self._store_loo(y - pred)

    def fit_residuals(self, y: Array, lower: Array, upper: Array) -> JackknifePlus:
        """Residual jackknife around existing bands: pred = mid, then exact LOO."""
        y = _as_1d(y)
        lower = _as_1d(lower)
        upper = _as_1d(upper)
        if lower.size == 1 and y.size != 1:
            lower = np.full(y.size, float(lower.item()))
        if upper.size == 1 and y.size != 1:
            upper = np.full(y.size, float(upper.item()))
        if not (y.size == lower.size == upper.size):
            raise ValueError("y, lower, and upper must have the same length")
        mask = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
        y_m, lo_m, hi_m = y[mask], lower[mask], upper[mask]
        if int(y_m.size) < 2:
            raise ValueError("Jackknife+ needs n>=2 for leave-one-out")
        exceed = np.maximum(np.maximum(lo_m - y_m, y_m - hi_m), 0.0)
        loc, scale = loo_mean_and_scale(y_m - 0.5 * (lo_m + hi_m))
        self.loo_loc_ = loc
        self.loo_scale_ = scale
        self.scores_ = exceed
        self.n_ = int(y_m.size)
        return self

    def _ready(self) -> bool:
        return self.loo_loc_ is not None and self.scores_ is not None and self.n_ >= 2

    def predict_sets(self, lower: Array, upper: Array) -> tuple[Array, Array]:
        lo = _as_1d(lower)
        hi = _as_1d(upper)
        if lo.size == 0:
            return lo.copy(), hi.copy()
        if lo.size != hi.size:
            raise ValueError("lower and upper must have the same length")
        if not self._ready():
            return np.full(lo.size, np.nan), np.full(hi.size, np.nan)
        scores = self.scores_
        if scores is None:
            raise RuntimeError("jackknife+ model has no fitted scores")
        qlo = lo[:, None] - scores[None, :]
        qhi = hi[:, None] + scores[None, :]
        return _jackknife_plus_quantiles(qlo, qhi, self.alpha)

    def predict_interval(
        self,
        x_mid: Array | float,
        scale: Array | float | None = None,
    ) -> tuple[Array, Array]:
        mid = np.atleast_1d(np.asarray(x_mid, dtype=float)).reshape(-1)
        if mid.size == 0:
            return mid.copy(), mid.copy()
        if not self._ready():
            return np.full(mid.size, np.nan), np.full(mid.size, np.nan)
        if scale is None:
            sc = np.ones(mid.size, dtype=float)
        else:
            sc = np.asarray(scale, dtype=float).reshape(-1)
            if sc.size == 1:
                sc = np.full(mid.size, float(sc.item()))
            elif sc.size != mid.size:
                raise ValueError("x_mid and scale must have the same length")
        sc = np.maximum(sc, 1e-12)
        loc = self.loo_loc_
        scores = self.scores_
        if loc is None or scores is None:
            raise RuntimeError("jackknife+ model has incomplete fitted state")
        half = scores[None, :] * sc[:, None]
        qlo = mid[:, None] + loc[None, :] - half
        qhi = mid[:, None] + loc[None, :] + half
        return _jackknife_plus_quantiles(qlo, qhi, self.alpha)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="conformal",
            name="jackknife_plus",
            version="v1",
            extra={
                "alpha": self.alpha,
                "n": self.n_,
                "z": self.z,
                "coverage_identity": "1-2*alpha",
                # Barber–Candès–Ramdas–Tibshirani 2021: marginal under exchangeability.
                # Not training-conditional (Bian–Barber 2023 caveat). Research-only.
                "coverage_guarantee_scope": "marginal_exchangeable",
                "coverage_guarantee_claim": (
                    "coverage floor is marginal under exchangeability; not training-conditional"
                ),
                "research_only": True,
            },
        )


def jackknife_plus_coverage_level(alpha: float) -> float:
    """Classic Jackknife+ finite-sample lower bound (1-2α). Not 1-α.

    Marginal under exchangeability (Barber–Candès–Ramdas–Tibshirani 2021);
    not a training-conditional guarantee (Bian–Barber 2023).
    """
    return 1.0 - 2.0 * float(alpha)
