"""Post-hoc recalibration of distributional forecasts (CRPS / quantile grid).

Post-hoc recalibration adjusts an already-fitted forecast with a small,
sample-based correction so the predictive distribution matches the
calibration empirical distribution — the calibration principle of Gneiting,
Balabdaoui & Raftery (2007, JRSS B 69(2):243–268,
https://doi.org/10.1111/j.1467-9868.2007.00587.x). Four estimators:

1. ``VarianceScalingGaussian`` — scalar scale correction for Gaussian
   location-scale forecasts, fitted by minimizing mean closed-form CRPS
   (variance scaling; Thorarinsdottir & Gneiting 2010, JRSS A 173(2):371–388,
   https://doi.org/10.1111/j.1467-985X.2009.00616.x).
2. ``VarianceScalingStudentT`` — same, for a location-scale Student-t with
   fixed degrees of freedom fitted at construction.
3. ``QuantileMappingCalibrator`` — per-level offset correction equal to the
   gap between the empirical quantile of ``y_cal`` and the mean predicted
   quantile, enforced monotone in the level direction by isotonic regression
   (statistical correction / quantile mapping; Déqué 2007, Global and
   Planetary Change 57(1–2):16–26,
   https://doi.org/10.1016/j.gloplacha.2006.11.030). Non-crossing of the
   transformed grid is guaranteed by rearrangement (Chernozhukov,
   Fernández-Val & Galichon 2010, Econometrica 78(3):1093–1145,
   https://doi.org/10.3982/ECTA7880) via ``rearrange_quantiles``.
4. ``IsotonicQuantileCalibrator`` — per-level isotonic regression of
   predicted-quantile value onto ``y_cal`` (isotonic recalibration in the
   spirit of Zadrozny & Elkan 2002, KDD'02, https://doi.org/10.1145/775047.775151,
   extended to quantile levels à la Meinshausen 2006, JMLR 7:983–999,
   https://jmlr.org/papers/v7/meinshausen06a.html). Non-crossing is enforced
   by cumulative maximization along the level axis.

Scores are proper (CRPS); no Sharpe/Sortino/P&L content. All estimators are
fail-closed: degenerate inputs raise ValueError rather than returning a
silent identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from quant_fund.metrics.scoring import (
    crps_gaussian,
    crps_student_t,
    rearrange_quantiles,
)

Array = NDArray[np.float64]

__all__ = [
    "LocationScaleParams",
    "VarianceScalingGaussian",
    "VarianceScalingStudentT",
    "QuantileMappingCalibrator",
    "IsotonicQuantileCalibrator",
]

_MIN_CAL = 30
_LOG_SCALE_BOUNDS: tuple[float, float] = (-3.0, 3.0)


@dataclass
class LocationScaleParams:
    """Recalibrated location-scale parameters (``df=None`` for Gaussian)."""

    mu: Array
    sigma: Array
    df: float | None = None


def _as_1d(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim > 1:
        raise ValueError(f"{name} must be 1d")
    return arr.reshape(-1)


def _require_same_length(*named: tuple[str, Array]) -> None:
    lengths = {name: arr.shape[0] for name, arr in named}
    if len(set(lengths.values())) > 1:
        parts = ", ".join(f"{k}={v}" for k, v in lengths.items())
        raise ValueError(f"length mismatch: {parts}")


def _require_finite(name: str, arr: Array) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")


def _validate_levels(levels: Array) -> Array:
    lv = _as_1d("levels", levels)
    if lv.size < 2:
        raise ValueError("need at least 2 levels")
    _require_finite("levels", lv)
    if np.any((lv <= 0.0) | (lv >= 1.0)):
        raise ValueError("levels must lie strictly inside (0, 1)")
    if np.any(np.diff(lv) <= 0.0):
        raise ValueError("levels must be strictly increasing")
    return lv


def _fit_calibration_rows(y_cal: Array) -> Array:
    y = _as_1d("y_cal", y_cal)
    _require_finite("y_cal", y)
    if y.size < _MIN_CAL:
        raise ValueError(f"need n_cal >= {_MIN_CAL} calibration rows, got {y.size}")
    return y


class _VarianceScalingBase:
    """Shared CRPS-minimizing scalar-scale fit on a bounded log-scale grid."""

    _df: float | None = None

    def __init__(self, log_bounds: tuple[float, float] = _LOG_SCALE_BOUNDS) -> None:
        lo, hi = float(log_bounds[0]), float(log_bounds[1])
        if not (np.isfinite(lo) and np.isfinite(hi)) or not lo < hi:
            raise ValueError("log_bounds must be finite and increasing")
        self.log_bounds = (lo, hi)
        self.scale_: float | None = None
        self.n_cal_: int = 0

    def _crps_mean(self, mu: Array, sigma: Array, y: Array, scale: float) -> float:
        raise NotImplementedError

    def fit(self, mu: Array, sigma: Array, y_cal: Array) -> _VarianceScalingBase:
        """Fit the scalar scale ``s`` minimizing mean CRPS of ``(mu, s·sigma)``.

        ``mu``/``sigma``/``y_cal`` must be finite 1d arrays of equal length
        with ``n_cal >= 30`` and ``sigma > 0``; optimization runs over
        ``log(s)`` in ``log_bounds`` with ``scipy.optimize.minimize_scalar``
        (bounded). Stores the result in ``scale_``.
        """
        m = _as_1d("mu", mu)
        s = _as_1d("sigma", sigma)
        y = _fit_calibration_rows(y_cal)
        _require_same_length(("mu", m), ("sigma", s), ("y_cal", y))
        _require_finite("mu", m)
        _require_finite("sigma", s)
        if np.any(s <= 0.0):
            raise ValueError("sigma must be positive")

        def objective(log_scale: float) -> float:
            return float(self._crps_mean(m, s, y, float(np.exp(log_scale))))

        result = minimize_scalar(objective, bounds=self.log_bounds, method="bounded")
        if not result.success or not np.isfinite(result.x):
            raise ValueError("scale optimization failed to converge")
        self.scale_ = float(np.exp(result.x))
        self.n_cal_ = int(y.size)
        return self

    def transform(self, mu: Array, sigma: Array) -> LocationScaleParams:
        """Return ``(mu, scale_·sigma)``; raises if not fitted."""
        if self.scale_ is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted: returning raw sigma as "
                "'calibrated' would masquerade an uncalibrated identity. "
                "Call fit() with >=30 finite rows first."
            )
        m = _as_1d("mu", mu)
        s = _as_1d("sigma", sigma)
        _require_same_length(("mu", m), ("sigma", s))
        _require_finite("mu", m)
        _require_finite("sigma", s)
        if np.any(s <= 0.0):
            raise ValueError("sigma must be positive")
        return LocationScaleParams(mu=m, sigma=self.scale_ * s, df=self._df)

    def fit_transform(self, mu: Array, sigma: Array, y_cal: Array) -> LocationScaleParams:
        """Fit on the calibration rows then recalibrate those same rows."""
        return self.fit(mu, sigma, y_cal).transform(mu, sigma)


class VarianceScalingGaussian(_VarianceScalingBase):
    """Scalar variance scaling for Gaussian forecasts via mean-CRPS fit.

    Cites the calibration principle (Gneiting, Balabdaoui & Raftery 2007)
    and variance scaling (Thorarinsdottir & Gneiting 2010). Exposes ``scale_``.
    """

    def _crps_mean(self, mu: Array, sigma: Array, y: Array, scale: float) -> float:
        return float(np.mean(crps_gaussian(y, mu, scale * sigma)))


class VarianceScalingStudentT(_VarianceScalingBase):
    """Scalar variance scaling for fixed-df Student-t location-scale forecasts.

    Degrees of freedom ``df`` are fixed at construction (``df > 2`` so the
    CRPS is finite; Jordan, Krüger & Lerch 2019 / scoringRules); only the
    scale is fitted. Same calibration principle as the Gaussian variant.
    """

    def __init__(self, df: float, log_bounds: tuple[float, float] = _LOG_SCALE_BOUNDS) -> None:
        super().__init__(log_bounds)
        nu = float(df)
        if not np.isfinite(nu) or nu <= 2.0:
            raise ValueError("df must be finite and > 2 (finite-variance CRPS)")
        self.df = nu
        self._df = nu

    def _crps_mean(self, mu: Array, sigma: Array, y: Array, scale: float) -> float:
        return float(np.mean(crps_student_t(y, mu, scale * sigma, nu=self.df)))


class QuantileMappingCalibrator:
    """Per-level offset correction with isotonic level-direction smoothing.

    For each level ``tau`` the fitted correction is

        offset(tau) = Quantile(y_cal, tau) − mean_i quantiles[i, tau],

    the empirical-vs-predicted quantile gap (quantile mapping / statistical
    correction; Déqué 2007), averaged across calibration points. The
    corrected curve ``mean_i quantiles[i, ·] + offset(·)`` is enforced
    nondecreasing in the level direction by ``sklearn.isotonic.
    IsotonicRegression``. ``transform`` adds the correction (linear
    interpolation over levels) and applies ``rearrange_quantiles`` so the
    output grid is provably non-crossing (Chernozhukov, Fernández-Val &
    Galichon 2010).
    """

    def __init__(self, levels: Array) -> None:
        self.levels_ = _validate_levels(levels)
        self.offsets_: Array | None = None
        self.n_cal_: int = 0

    def fit(self, quantiles: Array, y_cal: Array) -> QuantileMappingCalibrator:
        """Fit the monotone offset curve on ``(quantiles, y_cal)``.

        ``quantiles`` is ``(n_cal, q)`` aligned with ``levels_``; ``y_cal`` is
        ``(n_cal,)``. Requires ``n_cal >= 30`` and finite inputs.
        """
        q = np.asarray(quantiles, dtype=float)
        if q.ndim != 2 or q.shape[1] != self.levels_.size:
            raise ValueError(f"quantiles must be (n, {self.levels_.size}) matching levels")
        y = _fit_calibration_rows(y_cal)
        if q.shape[0] != y.size:
            raise ValueError("quantiles rows must align with y_cal")
        _require_finite("quantiles", q)
        emp = np.quantile(y, self.levels_)
        base = q.mean(axis=0)
        corrected = emp - base
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(self.levels_, corrected)
        self.offsets_ = np.asarray(iso.predict(self.levels_), dtype=float)
        self.n_cal_ = int(y.size)
        return self

    def _offset_at(self, levels: Array) -> Array:
        if self.offsets_ is None:
            raise RuntimeError(
                "QuantileMappingCalibrator is not fitted: returning raw "
                "quantiles as 'calibrated' would masquerade an uncalibrated "
                "identity. Call fit() with >=30 finite rows first."
            )
        lv = _validate_levels(levels)
        if lv.size != self.levels_.size or not np.allclose(lv, self.levels_):
            raise ValueError("transform levels must match the fitted levels")
        return np.interp(lv, self.levels_, self.offsets_)

    def transform(self, quantiles: Array, levels: Array | None = None) -> Array:
        """Apply the correction and rearrange to a non-crossing grid.

        ``levels`` defaults to the fitted levels; any custom levels must
        match them (the offset curve is evaluated by linear interpolation).
        """
        q = np.asarray(quantiles, dtype=float)
        if q.ndim != 2:
            raise ValueError("quantiles must be 2d")
        lv = self.levels_ if levels is None else _as_1d("levels", levels)
        if q.shape[1] != lv.size:
            raise ValueError("quantiles columns must match levels")
        _require_finite("quantiles", q)
        offset = self._offset_at(lv)
        corrected = q + offset.reshape(1, -1)
        return rearrange_quantiles(corrected)

    def fit_transform(self, quantiles: Array, y_cal: Array) -> Array:
        """Fit on the calibration rows then recalibrate those same rows."""
        return self.fit(quantiles, y_cal).transform(quantiles)


class IsotonicQuantileCalibrator:
    """Per-level isotonic regression of predicted-quantile value onto ``y_cal``.

    For each level ``tau`` a separate ``sklearn.isotonic.IsotonicRegression``
    (increasing, out-of-bounds clip) maps the predicted quantile value to the
    conditional empirical level of ``y_cal`` — isotonic recalibration in the
    spirit of Zadrozny & Elkan (2002) extended per quantile level à la
    Meinshausen (2006). ``transform`` applies each level's isotonic function
    and enforces non-crossing by cumulative maximization along the level
    axis.

    Out-of-sample protocol: the isotonic functions are fit on the first 70%
    of the calibration rows (a fixed temporal-style split); the remaining 30%
    select nothing — kept deliberately simple, so ``transform`` on held-out
    rows is a documented in-sample-style application of the fitted monotone
    maps, not a cross-validated estimate.
    """

    _FIT_FRACTION = 0.70

    def __init__(self, levels: Array) -> None:
        self.levels_ = _validate_levels(levels)
        self.isotonics_: list[IsotonicRegression] | None = None
        self.n_cal_: int = 0

    def fit(self, quantiles: Array, y_cal: Array) -> IsotonicQuantileCalibrator:
        """Fit one increasing isotonic map per level on the first 70% of rows.

        ``quantiles`` is ``(n_cal, q)`` aligned with ``levels_``; ``y_cal`` is
        ``(n_cal,)``. Requires ``n_cal >= 30`` and finite inputs.
        """
        q = np.asarray(quantiles, dtype=float)
        if q.ndim != 2 or q.shape[1] != self.levels_.size:
            raise ValueError(f"quantiles must be (n, {self.levels_.size}) matching levels")
        y = _fit_calibration_rows(y_cal)
        if q.shape[0] != y.size:
            raise ValueError("quantiles rows must align with y_cal")
        _require_finite("quantiles", q)
        n_fit = int(self._FIT_FRACTION * y.size)
        if n_fit < _MIN_CAL:
            raise ValueError(
                f"70% fit split leaves {n_fit} rows; need >= {_MIN_CAL} calibration rows"
            )
        q_fit, y_fit = q[:n_fit], y[:n_fit]
        isotonics: list[IsotonicRegression] = []
        for j in range(self.levels_.size):
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(q_fit[:, j], y_fit)
            isotonics.append(iso)
        self.isotonics_ = isotonics
        self.n_cal_ = int(y.size)
        return self

    def transform(self, quantiles: Array) -> Array:
        """Apply each level's isotonic map, then cumulative-max non-crossing."""
        if self.isotonics_ is None:
            raise RuntimeError(
                "IsotonicQuantileCalibrator is not fitted: returning raw "
                "quantiles as 'calibrated' would masquerade an uncalibrated "
                "identity. Call fit() with >=30 finite rows first."
            )
        q = np.asarray(quantiles, dtype=float)
        if q.ndim != 2 or q.shape[1] != self.levels_.size:
            raise ValueError(f"quantiles must be (n, {self.levels_.size}) matching levels")
        _require_finite("quantiles", q)
        corrected = np.empty_like(q)
        for j, iso in enumerate(self.isotonics_):
            corrected[:, j] = np.asarray(iso.predict(q[:, j]), dtype=float)
        return np.maximum.accumulate(corrected, axis=1)

    def fit_transform(self, quantiles: Array, y_cal: Array) -> Array:
        """Fit on the calibration rows then recalibrate those same rows."""
        return self.fit(quantiles, y_cal).transform(quantiles)
