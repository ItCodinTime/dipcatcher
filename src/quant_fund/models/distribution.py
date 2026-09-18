"""Return distribution: empirical, Gaussian, linear QR, tree quantiles."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import QuantileRegressor

from quant_fund.metrics.probability import pit_ks
from quant_fund.metrics.scoring import (
    coverage,
    crps_from_quantiles,
    pit_values,
    rearrange_quantiles,
)
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.ranking import _finite

NU_MIN = 3.0
NU_MAX = 30.0
_WRAPPEE_SCORE_TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class EmpiricalDistribution(JoblibMixin):
    def __init__(self, taus: list[float]) -> None:
        self.taus = taus
        self.q_: NDArray[np.float64] | None = None

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> EmpiricalDistribution:
        yy = y[np.isfinite(y)]
        self.q_ = np.quantile(yy, self.taus) if yy.size else np.zeros(len(self.taus))
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.q_ is None:
            raise RuntimeError("distribution model has not been fitted")
        return np.tile(self.q_, (x.shape[0], 1))

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="distribution", name="empirical", version="v1")


class GaussianDistribution(JoblibMixin):
    def __init__(self, taus: list[float]) -> None:
        from scipy.stats import norm

        self.taus = taus
        self.z = np.array([norm.ppf(t) for t in taus])
        self.mu = 0.0
        self.sig = 0.01
        self._fitted = False

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> GaussianDistribution:
        yy = y[np.isfinite(y)]
        self.mu = float(np.mean(yy)) if yy.size else 0.0
        self.sig = float(np.std(yy, ddof=1)) if yy.size > 1 else 0.01
        if not np.isfinite(self.sig) or self.sig <= 0.0:
            self.sig = 0.01
        self._fitted = True
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("distribution model has not been fitted")
        q = self.mu + self.sig * self.z
        return np.tile(q, (x.shape[0], 1))

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="distribution", name="gaussian", version="v1")


class ScaledGaussianDistribution(JoblibMixin):
    """Gaussian quantiles with PIT-safe scale (typically vol_20).

    Homoskedastic ``GaussianDistribution`` stays the misspecification diagnostic.
    This is the conditional competitor: q_τ(x) = μ + σ̂_z · scale(x) · z_τ.
    """

    def __init__(self, taus: list[float]) -> None:
        from scipy.stats import norm

        self.taus = taus
        self.z = np.array([norm.ppf(t) for t in taus])
        self.mu = 0.0
        self.z_sig = 1.0
        self._fitted = False

    def fit(
        self,
        y: NDArray[np.float64],
        scale: NDArray[np.float64],
        **kwargs: Any,
    ) -> ScaledGaussianDistribution:
        yy = np.asarray(y, dtype=float).reshape(-1)
        sc = np.maximum(np.asarray(scale, dtype=float).reshape(-1), 1e-8)
        if yy.size != sc.size:
            raise ValueError("y and scale must have the same length")
        mask = np.isfinite(yy) & np.isfinite(sc)
        yy, sc = yy[mask], sc[mask]
        self.mu = float(np.mean(yy)) if yy.size else 0.0
        z = (yy - self.mu) / sc if yy.size else np.array([0.0])
        self.z_sig = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
        if not np.isfinite(self.z_sig) or self.z_sig <= 0.0:
            self.z_sig = 1.0
        self._fitted = True
        return self

    def predict(self, scale: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("distribution model has not been fitted")
        sc = np.maximum(np.asarray(scale, dtype=float).reshape(-1), 1e-8)
        width = self.z_sig * sc
        return np.asarray(self.mu + width[:, None] * self.z[None, :], dtype=np.float64)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="distribution",
            name="scaled_gaussian",
            version="v1",
            extra={"z_sig": self.z_sig, "mu": self.mu},
        )


class ScaledEmpiricalDistribution(JoblibMixin):
    """Empirical standardized-residual distribution with PIT-safe scale.

    The conditional scale is supplied by a point-in-time covariate (normally
    ``vol_20``); only the training residual shape is learned. This is a
    non-parametric density competitor, not a claim of conditional coverage.
    """

    def __init__(self, taus: list[float]) -> None:
        self.taus = taus
        self.mu = 0.0
        self.z_quantiles = np.zeros(len(taus), dtype=float)
        self._fitted = False

    def fit(
        self,
        y: NDArray[np.float64],
        scale: NDArray[np.float64],
        **kwargs: Any,
    ) -> ScaledEmpiricalDistribution:
        yy, sc = _finite_y_scale(y, scale)
        self.mu = float(np.mean(yy)) if yy.size else 0.0
        z = (yy - self.mu) / sc if yy.size else np.array([], dtype=float)
        self.z_quantiles = (
            np.quantile(z, self.taus).astype(float)
            if z.size
            else np.zeros(len(self.taus), dtype=float)
        )
        self._fitted = True
        return self

    def predict(self, scale: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("distribution model has not been fitted")
        sc = np.maximum(np.asarray(scale, dtype=float).reshape(-1), 1e-8)
        return np.asarray(self.mu + sc[:, None] * self.z_quantiles[None, :], dtype=np.float64)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="distribution",
            name="scaled_empirical",
            version="v1",
            extra={"mu": self.mu, "n_quantiles": len(self.z_quantiles)},
        )


def _finite_y_scale(
    y: NDArray[np.float64], scale: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    yy = np.asarray(y, dtype=float).reshape(-1)
    sc = np.maximum(np.asarray(scale, dtype=float).reshape(-1), 1e-8)
    if yy.size != sc.size:
        raise ValueError("y and scale must have the same length")
    mask = np.isfinite(yy) & np.isfinite(sc)
    return yy[mask], sc[mask]


def _fit_student_z(z: NDArray[np.float64]) -> tuple[float, float]:
    """MLE of σ_z, ν for z ≈ σ_z t_ν. Clamp ν to [3, 30]; refit scale after clamp."""
    from scipy.stats import t as student_t

    z = np.asarray(z, dtype=float).reshape(-1)
    z = z[np.isfinite(z)]
    if z.size < 8:
        sig = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
        if not np.isfinite(sig) or sig <= 0.0:
            sig = 1.0
        return sig, 8.0
    import warnings

    df, sc = 8.0, float(np.std(z, ddof=1) or 1.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_hat, _loc, sc_hat = student_t.fit(z, floc=0.0)
        if np.isfinite(df_hat) and df_hat > 2.0:
            df = float(df_hat)
        if np.isfinite(sc_hat) and sc_hat > 0.0:
            sc = float(sc_hat)
    except (ValueError, RuntimeError, FloatingPointError):
        pass
    nu = float(np.clip(df if np.isfinite(df) else 8.0, NU_MIN, NU_MAX))
    if not np.isfinite(nu):
        nu = 8.0
    if abs(nu - df) > 1e-6 or not np.isfinite(sc) or sc <= 0.0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _df, _loc, sc2 = student_t.fit(z, floc=0.0, f0=nu)
            if np.isfinite(sc2) and sc2 > 0.0:
                sc = float(sc2)
        except (ValueError, RuntimeError, FloatingPointError):
            var = float(np.var(z, ddof=1)) if z.size > 1 else 1.0
            sc = float(np.sqrt(max(var * (nu - 2.0) / nu, 1e-12)))
    if not np.isfinite(sc) or sc <= 0.0:
        sc = 1.0
    return float(sc), nu


class ScaledStudentTDistribution(JoblibMixin):
    """Location-scale Student-t with PIT-safe scale (typically vol_20).

    Standardized residuals z = (y − μ) / scale. Fit σ_z and ν on z (ν clamped
    to [3, 30]). Quantiles: q_τ(x) = μ + σ̂_z · scale(x) · t_ν.ppf(τ).
    """

    def __init__(self, taus: list[float]) -> None:
        self.taus = taus
        self.mu = 0.0
        self.z_sig = 1.0
        self.nu = 8.0
        self._fitted = False

    def fit(
        self,
        y: NDArray[np.float64],
        scale: NDArray[np.float64],
        **kwargs: Any,
    ) -> ScaledStudentTDistribution:
        yy, sc = _finite_y_scale(y, scale)
        self.mu = float(np.mean(yy)) if yy.size else 0.0
        z = (yy - self.mu) / sc if yy.size else np.array([0.0])
        self.z_sig, self.nu = _fit_student_z(z)
        self._fitted = True
        return self

    def predict(self, scale: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("distribution model has not been fitted")
        from scipy.stats import t as student_t

        sc = np.maximum(np.asarray(scale, dtype=float).reshape(-1), 1e-8)
        z = np.array([student_t.ppf(t, self.nu) for t in self.taus])
        width = self.z_sig * sc
        return np.asarray(self.mu + width[:, None] * z[None, :], dtype=np.float64)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="distribution",
            name="scaled_student_t",
            version="v1",
            extra={"z_sig": self.z_sig, "mu": self.mu, "nu": self.nu},
        )


def select_wrappee_family_name(
    y_train: NDArray[np.float64],
    scale_train: NDArray[np.float64],
    y_holdout: NDArray[np.float64],
    scale_holdout: NDArray[np.float64],
    *,
    min_coverage: float = 0.85,
) -> str:
    """Score scaled-t vs scaled-gaussian on holdout; return family name only.

    Does **not** refit the operational ``taus`` model — selection freshness only.
    Use with ``fit_scaled_wrappee`` / ``resolve_wrappee_reselect_cached`` so an
    enlarged cal can change the family while a matching train-fit stays reusable.
    """
    score_taus = list(_WRAPPEE_SCORE_TAUS)
    gauss = ScaledGaussianDistribution(score_taus).fit(y_train, scale_train)
    student = ScaledStudentTDistribution(score_taus).fit(y_train, scale_train)
    qg = gauss.predict(scale_holdout)
    qt = student.predict(scale_holdout)
    yy = np.asarray(y_holdout, dtype=float).reshape(-1)
    lo_i, hi_i = 0, len(score_taus) - 1
    t_arr = np.array(score_taus)
    cov_t = coverage(yy, qt[:, lo_i], qt[:, hi_i])
    crps_g = crps_from_quantiles(yy, qg, t_arr)
    crps_t = crps_from_quantiles(yy, qt, t_arr)
    ks_g, ks_p_g = pit_ks(pit_values(yy, qg, t_arr))
    ks_t, ks_p_t = pit_ks(pit_values(yy, qt, t_arr))
    t_covers = bool(np.isfinite(cov_t) and cov_t >= min_coverage)
    t_better_crps = bool(np.isfinite(crps_t) and np.isfinite(crps_g) and crps_t < crps_g)
    t_better_pit = False
    if (
        np.isfinite(ks_p_t)
        and np.isfinite(ks_p_g)
        and ks_p_t > ks_p_g
        or (
            np.isfinite(ks_t)
            and np.isfinite(ks_g)
            and (not np.isfinite(ks_p_t) or not np.isfinite(ks_p_g) or ks_p_t == ks_p_g)
            and ks_t < ks_g
        )
    ):
        t_better_pit = True
    return "scaled_student_t" if t_covers and (t_better_crps or t_better_pit) else "scaled_gaussian"


def fit_scaled_wrappee(
    name: str,
    taus: list[float],
    y_train: NDArray[np.float64],
    scale_train: NDArray[np.float64],
) -> ScaledGaussianDistribution | ScaledStudentTDistribution:
    """Fit a named scaled wrappee family on train only (no holdout scoring)."""
    if name == "scaled_student_t":
        return ScaledStudentTDistribution(taus).fit(y_train, scale_train)
    if name == "scaled_gaussian":
        return ScaledGaussianDistribution(taus).fit(y_train, scale_train)
    raise ValueError(f"unknown scaled wrappee family: {name!r}")


def select_scaled_wrappee(
    taus: list[float],
    y_train: NDArray[np.float64],
    scale_train: NDArray[np.float64],
    y_holdout: NDArray[np.float64],
    scale_holdout: NDArray[np.float64],
    *,
    nominal_coverage: float = 0.90,
    min_coverage: float = 0.85,
) -> tuple[str, ScaledGaussianDistribution | ScaledStudentTDistribution]:
    """Pick scaled-t when it improves holdout CRPS or PIT KS without undercovering.

    The returned model is always refit on ``y_train`` with ``taus``. Homoskedastic
    Gaussian is not a candidate; this chooses among scaled location-scale wrappees.
    Composition of ``select_wrappee_family_name`` + ``fit_scaled_wrappee``.
    """
    _ = nominal_coverage
    name = select_wrappee_family_name(
        y_train,
        scale_train,
        y_holdout,
        scale_holdout,
        min_coverage=min_coverage,
    )
    return name, fit_scaled_wrappee(name, taus, y_train, scale_train)


def fit_operational_wrappee(
    taus: list[float],
    y_train: NDArray[np.float64],
    scale_train: NDArray[np.float64],
    *,
    nominal_coverage: float = 0.90,
    min_coverage: float = 0.85,
) -> tuple[str, ScaledGaussianDistribution | ScaledStudentTDistribution]:
    """Select wrappee on a chronological inner holdout, then refit on all of train."""
    yy, sc = _finite_y_scale(y_train, scale_train)
    n = int(yy.size)
    cut = max(int(0.8 * n), 8)
    if n < 24 or n - cut < 8:
        cut = max(n // 2, 8)
    if n - cut < 8 or cut < 8:
        return "scaled_gaussian", ScaledGaussianDistribution(taus).fit(yy, sc)
    name, _ = select_scaled_wrappee(
        taus,
        yy[:cut],
        sc[:cut],
        yy[cut:],
        sc[cut:],
        nominal_coverage=nominal_coverage,
        min_coverage=min_coverage,
    )
    if name == "scaled_student_t":
        return name, ScaledStudentTDistribution(taus).fit(yy, sc)
    return name, ScaledGaussianDistribution(taus).fit(yy, sc)


class LinearQuantileDistribution(JoblibMixin):
    def __init__(self, taus: list[float]) -> None:
        self.taus = taus
        self.models = [QuantileRegressor(quantile=t, alpha=1e-4, solver="highs") for t in taus]
        self._fitted = False

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> LinearQuantileDistribution:
        xx, yy, _ = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("LinearQuantileDistribution requires at least one finite sample")
        for m in self.models:
            m.fit(xx, yy)
        self._fitted = True
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("distribution model has not been fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        cols = [m.predict(x) for m in self.models]
        return np.column_stack(cols)

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="distribution", name="linear_qr", version="v1")


class TreeQuantileDistribution(JoblibMixin):
    def __init__(self, taus: list[float], backend: str = "lightgbm", seed: int = 42) -> None:
        self.taus = taus
        self.backend = backend
        self.models: list[Any] = []
        self.seed = seed
        self._fitted = False

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> TreeQuantileDistribution:
        xx, yy, _ = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("TreeQuantileDistribution requires at least one finite sample")
        self.models = []
        for tau in self.taus:
            if self.backend == "xgboost":
                from xgboost import XGBRegressor

                m = XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=tau,
                    n_estimators=60,
                    max_depth=3,
                    n_jobs=1,
                    random_state=self.seed,
                    verbosity=0,
                )
            else:
                from lightgbm import LGBMRegressor

                m = LGBMRegressor(
                    objective="quantile",
                    alpha=tau,
                    n_estimators=60,
                    num_leaves=15,
                    random_state=self.seed,
                    verbosity=-1,
                )
            m.fit(xx, yy)
            self.models.append(m)
        self._fitted = True
        return self

    def predict(self, x: NDArray[np.float64], rearrange: bool = False) -> NDArray[np.float64]:
        if not self._fitted or not self.models:
            raise RuntimeError("distribution model has not been fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        q = np.column_stack([m.predict(x) for m in self.models])
        if rearrange:
            q = rearrange_quantiles(q)
        return q

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="distribution", name=f"{self.backend}_quantile", version="v1")
