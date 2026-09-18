"""K-fold CV+ / CV-minmax conformal wrapper.

Barber, Candès, Ramdas, Tibshirani (2021). K-fold residual scores replace
leave-one-out. Classic CV+ (eq. 11) has finite-sample coverage ≥ 1-2α − ε_n
(Theorem 4). The minmax interval on the same fold predictors (eq. 10 / Theorem 3)
has coverage ≥ 1-α. Default aggregation is minmax so ``CVPlus`` documents 1-α.
Those floors are **marginal under exchangeability**, not training-conditional
(Bian & Barber 2023 caveat). Does not reimplement QR. Wraps a residual /
quantile predictor, or a K-fold mean±z Gaussian band.

Optional JAW (weighted CV+ ensemble; Tibshirani, Barber, Candès, Ramdas 2019
weights) is 1-2α under covariate shift, not 1-α.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.metrics.conformal import conformal_quantile, cqr_scores, expand_interval
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.weighted_conformal import weighted_conformal_quantile

Array = NDArray[np.float64]
_AGGREGATIONS = frozenset({"minmax", "plus", "jaw"})


def _as_1d(values: Array) -> Array:
    return np.asarray(values, dtype=float).reshape(-1)


def _align_with_dates(
    y: Array,
    other: Array,
    name: str,
    dates: Array | None,
) -> tuple[Array, Array, Array | None]:
    y = _as_1d(y)
    extra = np.asarray(other, dtype=float).reshape(-1)
    if extra.size == 1 and y.size != 1:
        extra = np.full(y.size, float(extra.item()))
    if y.size != extra.size:
        raise ValueError(f"y and {name} must have the same length")
    dates_arr: Array | None = None
    if dates is not None:
        dates_arr = np.asarray(dates).reshape(-1)
        if dates_arr.size != y.size:
            raise ValueError("y and dates must have the same length")
    mask = np.isfinite(y) & np.isfinite(extra)
    if dates_arr is None:
        return y[mask], extra[mask], None
    return y[mask], extra[mask], dates_arr[mask]


def _contiguous_fold_ids(n: int, n_folds: int) -> Array:
    """Nearly equal contiguous blocks. Never empty when n >= n_folds."""
    if n < n_folds:
        raise ValueError("n must be >= n_folds")
    return np.minimum(n_folds - 1, (np.arange(n, dtype=int) * n_folds) // n)


def assign_cv_folds(n: int, n_folds: int, dates: Array | None = None) -> Array:
    """K-fold ids. Dates group by time; rows on one date share a fold.

    Unique dates are ordered and split into contiguous time blocks. Panel rows
    are never stacked into iid folds when ``dates`` is passed.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if dates is None:
        return _contiguous_fold_ids(n, n_folds)
    d = np.asarray(dates).reshape(-1)
    if d.size != n:
        raise ValueError("dates must have the same length as y")
    uniq, inv = np.unique(d, return_inverse=True)
    n_t = int(uniq.size)
    if n_t < n_folds:
        raise ValueError("unique dates must be >= n_folds")
    return _contiguous_fold_ids(n_t, n_folds)[inv]


def kfold_mean_and_scale(residual: Array, fold_id: Array) -> tuple[Array, Array]:
    """Complement-fold mean and unbiased std, repeated onto each in-fold row.

    For i in fold k, μ_{-k} and σ_{-k} use residual[fold != k] only.
    Complement size 1 → scale 0 (undefined std).
    """
    r = np.asarray(residual, dtype=float).reshape(-1)
    f = np.asarray(fold_id).reshape(-1)
    if r.size != f.size:
        raise ValueError("residual and fold_id must have the same length")
    loc = np.empty(r.size, dtype=float)
    scale = np.empty(r.size, dtype=float)
    for k in np.unique(f):
        in_fold = f == k
        out = r[~in_fold]
        m = int(out.size)
        if m < 1:
            raise ValueError("each fold must leave at least one complement point")
        loc[in_fold] = float(np.mean(out))
        if m < 2:
            scale[in_fold] = 0.0
        else:
            scale[in_fold] = float(np.std(out, ddof=1))
    return loc, scale


def _plus_quantiles(lo_ens: Array, hi_ens: Array, alpha: float) -> tuple[Array, Array]:
    """Paper (11): q^- of {μ_{-k(i)}-R_i} and q^+ of {μ_{-k(i)}+R_i}."""
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


def _minmax_interval(qlo: Array, qhi: Array, scores: Array, alpha: float) -> tuple[Array, Array]:
    """Paper (10) on K-fold predictors: min μ_{-k} − q^+{R}, max μ_{-k} + q^+{R}."""
    qhat = conformal_quantile(scores, alpha)
    finite_lo = np.where(np.isfinite(qlo), qlo, np.inf)
    finite_hi = np.where(np.isfinite(qhi), qhi, -np.inf)
    has = np.any(np.isfinite(qlo) & np.isfinite(qhi), axis=1)
    out_lo = np.min(finite_lo, axis=1) - qhat
    out_hi = np.max(finite_hi, axis=1) + qhat
    return np.where(has, out_lo, np.nan), np.where(has, out_hi, np.nan)


def _jaw_quantiles(
    lo_ens: Array, hi_ens: Array, weights: Array, alpha: float
) -> tuple[Array, Array]:
    """Weighted CV+ ensemble quantiles (JAW). Uniform weights recover plus."""
    w = _as_1d(weights)
    if w.size != lo_ens.shape[1]:
        raise ValueError("weights must have one entry per training residual")
    m = int(lo_ens.shape[0])
    out_lo = np.empty(m, dtype=float)
    out_hi = np.empty(m, dtype=float)
    for i in range(m):
        row_lo = lo_ens[i]
        row_hi = hi_ens[i]
        finite = np.isfinite(row_lo) & np.isfinite(row_hi) & np.isfinite(w) & (w > 0)
        if not bool(np.any(finite)):
            out_lo[i] = np.nan
            out_hi[i] = np.nan
            continue
        out_lo[i] = -weighted_conformal_quantile(-row_lo[finite], w[finite], alpha)
        out_hi[i] = weighted_conformal_quantile(row_hi[finite], w[finite], alpha)
    return out_lo, out_hi


class CVPlus(JoblibMixin):
    """K-fold conformal wrapper. Default minmax guarantee is 1-α, not 1-2α."""

    def __init__(
        self,
        alpha: float = 0.10,
        n_folds: int = 5,
        aggregation: str = "minmax",
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if int(n_folds) < 2:
            raise ValueError("n_folds must be >= 2")
        agg = str(aggregation)
        if agg not in _AGGREGATIONS:
            raise ValueError("aggregation must be 'minmax', 'plus', or 'jaw'")
        self.alpha = float(alpha)
        self.n_folds = int(n_folds)
        self.aggregation = agg
        self.z = float(norm.ppf(1.0 - self.alpha / 2.0))
        self.fold_loc_: Array | None = None
        self.fold_scale_: Array | None = None
        self.scores_: Array | None = None
        self.fold_id_: Array | None = None
        self.n_ = 0

    def _store_folds(self, residual: Array, dates: Array | None = None) -> CVPlus:
        n = int(residual.size)
        if n == 0:
            raise ValueError("CV+ needs a non-empty sample")
        fold_id = assign_cv_folds(n, self.n_folds, dates)
        loc, scale = kfold_mean_and_scale(residual, fold_id)
        qlo = loc - self.z * scale
        qhi = loc + self.z * scale
        self.fold_loc_ = loc
        self.fold_scale_ = scale
        self.scores_ = cqr_scores(residual, qlo, qhi)
        self.fold_id_ = fold_id
        self.n_ = n
        return self

    def fit(
        self,
        y: Array,
        pred: Array | None = None,
        lower: Array | None = None,
        upper: Array | None = None,
        dates: Array | None = None,
    ) -> CVPlus:
        """K-fold mean±z band on residuals y-pred, or on y-mid if bands are given."""
        y = _as_1d(y)
        if lower is not None and upper is not None:
            lower_a = _as_1d(lower)
            upper_a = _as_1d(upper)
            if lower_a.size == 1 and y.size != 1:
                lower_a = np.full(y.size, float(lower_a.item()))
            if upper_a.size == 1 and y.size != 1:
                upper_a = np.full(y.size, float(upper_a.item()))
            if not (y.size == lower_a.size == upper_a.size):
                raise ValueError("y, lower, and upper must have the same length")
            dates_arr: Array | None = None
            if dates is not None:
                dates_arr = np.asarray(dates).reshape(-1)
                if dates_arr.size != y.size:
                    raise ValueError("y and dates must have the same length")
            mask = np.isfinite(y) & np.isfinite(lower_a) & np.isfinite(upper_a)
            residual = y[mask] - 0.5 * (lower_a[mask] + upper_a[mask])
            return self._store_folds(residual, None if dates_arr is None else dates_arr[mask])
        if pred is None:
            pred = np.zeros_like(y)
        y, pred, dates_f = _align_with_dates(y, pred, "pred", dates)
        return self._store_folds(y - pred, dates_f)

    def _ready(self) -> bool:
        return self.fold_loc_ is not None and self.scores_ is not None and self.n_ >= 2

    def _aggregate(
        self,
        qlo: Array,
        qhi: Array,
        weights: Array | None,
    ) -> tuple[Array, Array]:
        loc_scores = self.scores_
        if loc_scores is None:
            raise RuntimeError("CV+ model has no fitted scores")
        how = "jaw" if weights is not None else self.aggregation
        if how == "minmax":
            return _minmax_interval(qlo, qhi, loc_scores, self.alpha)
        lo_ens, hi_ens = expand_interval(qlo, qhi, loc_scores)
        if how == "jaw":
            w = np.ones(self.n_, dtype=float) if weights is None else weights
            return _jaw_quantiles(lo_ens, hi_ens, w, self.alpha)
        return _plus_quantiles(lo_ens, hi_ens, self.alpha)

    def predict_sets(
        self,
        lower: Array,
        upper: Array,
        weights: Array | None = None,
    ) -> tuple[Array, Array]:
        lo = _as_1d(lower)
        hi = _as_1d(upper)
        if lo.size == 0:
            return lo.copy(), hi.copy()
        if lo.size != hi.size:
            raise ValueError("lower and upper must have the same length")
        if not self._ready():
            return np.full(lo.size, np.nan), np.full(hi.size, np.nan)
        loc = self.fold_loc_
        if loc is None:
            raise RuntimeError("CV+ model has no fitted fold locations")
        qlo = lo[:, None] + loc[None, :]
        qhi = hi[:, None] + loc[None, :]
        return self._aggregate(qlo, qhi, weights)

    def predict_interval(
        self,
        x_mid: Array | float,
        scale: Array | float | None = None,
        weights: Array | None = None,
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
        loc = self.fold_loc_
        band = self.fold_scale_
        if loc is None or band is None:
            raise RuntimeError("CV+ model has no fitted fold scale")
        half = self.z * band[None, :] * sc[:, None]
        qlo = mid[:, None] + loc[None, :] - half
        qhi = mid[:, None] + loc[None, :] + half
        return self._aggregate(qlo, qhi, weights)

    def metadata(self) -> ModelMeta:
        identity = "1-alpha" if self.aggregation == "minmax" else "1-2*alpha"
        return ModelMeta(
            family="conformal",
            name="cv_plus",
            version="v1",
            extra={
                "alpha": self.alpha,
                "n_folds": self.n_folds,
                "n": self.n_,
                "z": self.z,
                "aggregation": self.aggregation,
                "coverage_identity": identity,
                # Barber–Candès–Ramdas–Tibshirani 2021: marginal under exchangeability
                # (minmax 1-α / plus|JAW 1-2α). Not training-conditional (Bian–Barber 2023).
                "coverage_guarantee_scope": "marginal_exchangeable",
                "coverage_guarantee_claim": (
                    "coverage floor is marginal under exchangeability; not training-conditional"
                ),
                "research_only": True,
            },
        )


def cv_plus_coverage_level(alpha: float, aggregation: str = "minmax") -> float:
    """Finite-sample lower bound. minmax is 1-α; plus and JAW are 1-2α.

    Marginal under exchangeability (Barber–Candès–Ramdas–Tibshirani 2021;
    aggregation-specific); not a training-conditional guarantee (Bian–Barber 2023).
    """
    if str(aggregation) == "minmax":
        return 1.0 - float(alpha)
    return 1.0 - 2.0 * float(alpha)
