"""Book-level conformal sets for the scalar r_p = w^T r.

One interval per date. Names on the same date are never stacked as if they
were i.i.d. scores. See docs/decisions/020-portfolio-conformal.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.metrics.conformal import (
    conformal_quantile,
    cqr_scores,
    expand_interval,
    set_metrics,
)
from quant_fund.models.base import JoblibMixin, ModelMeta

Array = NDArray[np.float64]
DateLike = Any
Panel = Mapping[DateLike, Array] | Array | Sequence[Array]


def portfolio_return(weights: Array | float, returns: Array | float) -> float:
    """Nan-safe book return r_p = w^T r.

    Names with a non-finite weight or return are dropped. Remaining weights
    are not renormalized.
    """
    w = np.asarray(weights, dtype=float).ravel()
    r = np.asarray(returns, dtype=float).ravel()
    if w.size != r.size:
        raise ValueError("weights and returns must have the same length")
    mask = np.isfinite(w) & np.isfinite(r)
    if not bool(mask.any()):
        return float("nan")
    return float(np.dot(w[mask], r[mask]))


def historical_port_vol(r_p: Array) -> float:
    """Sample standard deviation of realized book returns."""
    s = np.asarray(r_p, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return 1e-8
    return float(np.std(s, ddof=1))


def gaussian_band(mu: float, sigma: float, alpha: float) -> tuple[float, float]:
    """Symmetric Gaussian interval from a location-scale pair."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    sig = max(float(sigma), 1e-12)
    return float(mu) - z * sig, float(mu) + z * sig


def _chrono_split(n: int) -> tuple[slice, slice, slice]:
    """Chronological train / calibration / test. Calibration is never test."""
    if n < 30:
        a, b = max(n // 3, 1), max(2 * n // 3, 2)
        return slice(0, a), slice(a, b), slice(b, n)
    a = max(int(0.5 * n), 8)
    b = max(int(0.7 * n), a + 5)
    return slice(0, a), slice(a, b), slice(b, n)


def _ordered_keys(keys: Sequence[DateLike]) -> list[DateLike]:
    try:
        return sorted(keys)
    except TypeError:
        return sorted(keys, key=lambda d: str(d))


def _as_date_keys(dates: Sequence[DateLike]) -> list[str]:
    out: list[str] = []
    for d in dates:
        if isinstance(d, np.datetime64):
            out.append(str(d))
        elif hasattr(d, "isoformat"):
            out.append(d.isoformat())
        else:
            out.append(str(d))
    return out


def _stack_panel(panel: Panel, dates: Sequence[DateLike] | None) -> tuple[list[DateLike], Array]:
    if isinstance(panel, Mapping):
        keys = _ordered_keys(list(panel.keys()))
        if dates is not None:
            keys = [d for d in dates if d in panel]
        rows = [np.asarray(panel[d], dtype=float).ravel() for d in keys]
        if not rows:
            raise ValueError("weights_by_date / returns_by_date is empty")
        width = max(int(row.size) for row in rows)
        mat = np.full((len(rows), width), np.nan, dtype=float)
        for i, row in enumerate(rows):
            mat[i, : row.size] = row
        return keys, mat
    arr = np.asarray(panel, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("panel must be a date-map or a (n_dates, n_names) array")
    if dates is None:
        keys = list(range(int(arr.shape[0])))
    else:
        keys = list(dates)
        if len(keys) != int(arr.shape[0]):
            raise ValueError("dates must align with the panel rows")
    return keys, arr


def _align_vol(
    vol: Mapping[DateLike, float] | Array | float | None,
    dates: Sequence[DateLike],
    n: int,
) -> Array | None:
    if vol is None:
        return None
    if np.isscalar(vol):
        scalar = float(np.asarray(vol).reshape(-1)[0])
        return np.full(n, scalar, dtype=np.float64)
    if isinstance(vol, Mapping):
        return np.asarray([float(vol[d]) if d in vol else np.nan for d in dates], dtype=float)
    scale = np.asarray(vol, dtype=float).ravel()
    if scale.size == 1:
        return np.full(n, float(scale[0]), dtype=float)
    if scale.size != n:
        raise ValueError("vol must be scalar or align with dates")
    return scale


def _aligned_panels(
    weights_by_date: Panel,
    returns_by_date: Panel,
    dates: Sequence[DateLike] | None = None,
) -> tuple[list[DateLike], Array, Array]:
    w_dates, w = _stack_panel(weights_by_date, dates)
    r_dates, r = _stack_panel(returns_by_date, dates)
    if isinstance(weights_by_date, Mapping) and isinstance(returns_by_date, Mapping):
        common = [d for d in w_dates if d in set(r_dates)]
        w_idx = {d: i for i, d in enumerate(w_dates)}
        r_idx = {d: i for i, d in enumerate(r_dates)}
        w = w[[w_idx[d] for d in common]]
        r = r[[r_idx[d] for d in common]]
        w_dates = common
    elif w_dates != r_dates:
        if len(w_dates) != len(r_dates):
            raise ValueError("weights_by_date and returns_by_date must walk the same dates")
        w_dates = r_dates
    if w.shape[0] != r.shape[0]:
        raise ValueError("weights_by_date and returns_by_date must have the same n_dates")
    if w.shape[1] != r.shape[1]:
        width = max(int(w.shape[1]), int(r.shape[1]))
        w = np.pad(w, ((0, 0), (0, width - w.shape[1])), constant_values=np.nan)
        r = np.pad(r, ((0, 0), (0, width - r.shape[1])), constant_values=np.nan)
    return w_dates, w, r


class SplitPortfolioCQR(JoblibMixin):
    """Split CQR on the book scalar r_p. Optional portfolio-vol scale."""

    def __init__(self, alpha: float = 0.10) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.qhat = 0.0
        self.scores_: Array | None = None

    def calibrate(
        self,
        r_p: Array,
        lower: Array,
        upper: Array,
        scale: Array | None = None,
    ) -> SplitPortfolioCQR:
        s = cqr_scores(r_p, lower, upper, scale)
        self.scores_ = s
        self.qhat = conformal_quantile(s, self.alpha)
        return self

    def predict_sets(
        self,
        lower: Array,
        upper: Array,
        scale: Array | None = None,
    ) -> tuple[Array, Array]:
        return expand_interval(lower, upper, self.qhat, scale)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="conformal",
            name="split_portfolio_cqr",
            version="v1",
            extra={"alpha": self.alpha, "qhat": self.qhat},
        )


@dataclass(frozen=True)
class PortfolioSets:
    dates: list[str]
    r_p: Array
    lower: Array
    upper: Array
    qhat: float
    alpha: float
    n_cal: int
    test_mask: Array
    scale: Array | None = None
    scores: Array | None = None


def sets_by_date(
    weights_by_date: Panel,
    returns_by_date: Panel,
    alpha: float = 0.10,
    vol: Mapping[DateLike, float] | Array | float | None = None,
    dates: Sequence[DateLike] | None = None,
) -> PortfolioSets:
    """One conformal set per date for r_p = w^T r.

    Walks dates in order. Each date is a single conformal observation. The
    base interval is a Gaussian band from historical (train-window) portfolio
    volatility. Optional ``vol`` is a PIT-safe portfolio scale for
    normalized CQR.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    keys, w, r = _aligned_panels(weights_by_date, returns_by_date, dates)
    n = int(w.shape[0])
    if n == 0:
        raise ValueError("no dates to walk")
    r_p = np.asarray(
        [portfolio_return(w[i], r[i]) for i in range(n)],
        dtype=float,
    )
    scale = _align_vol(vol, keys, n)
    tr, cal, te = _chrono_split(n)
    mu = float(np.nanmean(r_p[tr])) if int(np.isfinite(r_p[tr]).sum()) else 0.0
    sigma = historical_port_vol(r_p[tr])
    lo_b, hi_b = gaussian_band(mu, sigma, alpha)
    lower = np.full(n, lo_b, dtype=float)
    upper = np.full(n, hi_b, dtype=float)
    sc_cal = None if scale is None else scale[cal]
    cqr = SplitPortfolioCQR(alpha).calibrate(r_p[cal], lower[cal], upper[cal], sc_cal)
    lo, hi = cqr.predict_sets(lower, upper, scale)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[te] = True
    scores = cqr.scores_
    n_cal = int(np.isfinite(scores).sum()) if scores is not None else 0
    return PortfolioSets(
        dates=_as_date_keys(keys),
        r_p=r_p,
        lower=lo,
        upper=hi,
        qhat=float(cqr.qhat),
        alpha=float(alpha),
        n_cal=n_cal,
        test_mask=test_mask,
        scale=scale,
        scores=scores,
    )


def _synthetic_book(
    n_dates: int,
    n_names: int,
    seed: int,
    scale: float = 0.02,
) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    weights = np.full((n_dates, n_names), 1.0 / n_names, dtype=float)
    returns = rng.normal(0.0, scale, size=(n_dates, n_names))
    return weights, returns


def bench_portfolio_cqr(
    weights_by_date: Panel | None = None,
    returns_by_date: Panel | None = None,
    alpha: float = 0.10,
    vol: Mapping[DateLike, float] | Array | float | None = None,
    *,
    n_dates: int = 400,
    n_names: int = 8,
    seed: int = 7,
) -> dict[str, float]:
    """Coverage and mean width of book-level CQR. No Sharpe."""
    if (weights_by_date is None) ^ (returns_by_date is None):
        raise ValueError("pass both weights_by_date and returns_by_date, or neither")
    if weights_by_date is None or returns_by_date is None:
        weights_by_date, returns_by_date = _synthetic_book(n_dates, n_names, seed)
    sets = sets_by_date(weights_by_date, returns_by_date, alpha=alpha, vol=vol)
    y = sets.r_p[sets.test_mask]
    lo = sets.lower[sets.test_mask]
    hi = sets.upper[sets.test_mask]
    metrics = set_metrics(y, lo, hi)
    return {
        "coverage": metrics.coverage,
        "mean_width": metrics.mean_width,
        "n_dates": float(metrics.n),
        "alpha": float(alpha),
        "qhat": float(sets.qhat),
        "n_cal": float(sets.n_cal),
    }
