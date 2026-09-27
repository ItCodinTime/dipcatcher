"""Vol-scaled h-step return-distribution challenger (ULTRAPLAN P1.10)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.scoring import rearrange_quantiles
from quant_fund.models.base import JoblibMixin, ModelMeta

NU_MIN = 3.0
NU_MAX = 30.0
MIN_OBS_BUFFER = 30


def _fit_t_nu(z: NDArray[np.float64]) -> float:
    """MLE degrees-of-freedom for z ~ t_nu (loc fixed at 0); clip to [NU_MIN, NU_MAX]."""
    import warnings

    from scipy.stats import t as student_t

    z = np.asarray(z, dtype=float).reshape(-1)
    z = z[np.isfinite(z)]
    if z.size < 8:
        return 8.0
    df = 8.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_hat, _loc, _sc = student_t.fit(z, floc=0.0)
        if np.isfinite(df_hat) and df_hat > 2.0:
            df = float(df_hat)
    except (ValueError, RuntimeError, FloatingPointError):
        pass
    nu = float(np.clip(df, NU_MIN, NU_MAX))
    return nu if np.isfinite(nu) else 8.0


def _overlap_sums(y: NDArray[np.float64], h: int) -> NDArray[np.float64]:
    """Overlapping h-step sums y_h[t] = sum_{i<h} y[t+i]; returns n - h + 1 sums."""
    c = np.empty(y.size + 1, dtype=float)
    c[0] = 0.0
    np.cumsum(y, out=c[1:])
    return np.asarray(c[h:] - c[:-h], dtype=float)


class HStepScaledDistribution(JoblibMixin):
    """Vol-scaled h-step distribution challenger (dip_hstep).

    Unconditional head: ``x`` is ignored and ``predict`` tiles one quantile
    row. Rows of ``y`` arrive in panel (date-major) order; the overlapping
    h-step sums treat them as a single time-ordered series, so on a
    multi-asset window the sums can straddle asset boundaries — an honest
    documented limitation, not hidden cleaning.

    Per horizon ``h`` in ``horizons`` two constructions are emitted:

    * ``student_t`` — iid location-scale h-step sum: ``mu_h = h * mu_1``,
      ``sigma_h = sigma_1 * sqrt(h)``, Student-t shape whose nu is MLE-fit
      on the window and capped to [NU_MIN, NU_MAX]. t quantiles are
      normalized to unit variance so sigma_1 stays the window std:
      ``q = h*mu_1 + sigma_1*sqrt(h) * t_nu.ppf(tau) / sqrt(nu/(nu-2))``.
    * ``empirical`` — np.quantile of the in-window overlapping sums
      ``y_h[t] = sum_{i<h} y[t+i]`` (n - h + 1 sums; overlapping-bootstrap
      analog of the h-step return distribution).

    ``predict`` output has ``2 * n_taus * n_horizons`` columns — one block
    per horizon (in ``horizons`` order) of ``[student_t(taus) |
    empirical(taus)]``: for block index b at ``h = horizons[b]``,
    ``cols[2*b*T : 2*b*T + T]`` are the student_t quantiles and
    ``cols[(2*b+1)*T : (2*b+2)*T]`` the empirical ones, ``T = len(taus)``.
    Each construction row is rearranged monotone. Horizons come from the
    constructor (default (1, 5, 20)); config plumbing is intentionally absent.
    """

    def __init__(self, taus: list[float], horizons: tuple[int, ...] = (1, 5, 20)) -> None:
        tt = np.asarray(taus, dtype=float).reshape(-1)
        if (
            tt.size == 0
            or not np.isfinite(tt).all()
            or np.any(tt <= 0.0)
            or np.any(tt >= 1.0)
            or np.any(np.diff(tt) <= 0.0)
        ):
            raise ValueError("taus must be non-empty and strictly increasing in (0, 1)")
        hs_raw = list(horizons)
        if not hs_raw or any(int(h) != h or int(h) < 1 for h in hs_raw):
            raise ValueError("horizons must be non-empty positive integers")
        self.taus = [float(t) for t in tt]
        self.horizons = tuple(sorted({int(h) for h in hs_raw}))
        self.mu_ = 0.0
        self.sig_ = 0.0
        self.nu_ = 0.0
        self.q_: NDArray[np.float64] | None = None
        self.q_student_: NDArray[np.float64] | None = None  # (n_horizons, n_taus)
        self.q_emp_: NDArray[np.float64] | None = None  # (n_horizons, n_taus)

    def _unit_t_ppf(self) -> NDArray[np.float64]:
        from scipy.stats import t as student_t

        tt = np.asarray(self.taus, dtype=float)
        return np.asarray(
            student_t.ppf(tt, self.nu_) / np.sqrt(self.nu_ / (self.nu_ - 2.0)),
            dtype=float,
        )

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> HStepScaledDistribution:
        yy = np.asarray(y, dtype=float).reshape(-1)
        yy = yy[np.isfinite(yy)]
        max_h = self.horizons[-1]
        if yy.size < MIN_OBS_BUFFER + max_h:
            raise ValueError(
                f"HStepScaledDistribution requires >= {MIN_OBS_BUFFER + max_h} finite observations"
            )
        self.mu_ = float(yy.mean())
        s = float(yy.std(ddof=1))
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("HStepScaledDistribution requires positive window variance")
        self.sig_ = s
        self.nu_ = _fit_t_nu((yy - self.mu_) / s)
        tt = np.asarray(self.taus, dtype=float)
        t_row = self._unit_t_ppf()
        q_st = np.empty((len(self.horizons), tt.size))
        q_emp = np.empty((len(self.horizons), tt.size))
        for b, h in enumerate(self.horizons):
            q_st[b] = h * self.mu_ + s * np.sqrt(h) * t_row
            q_emp[b] = np.quantile(_overlap_sums(yy, h), tt)
        self.q_student_ = np.asarray(rearrange_quantiles(q_st), dtype=float)
        self.q_emp_ = np.asarray(rearrange_quantiles(q_emp), dtype=float)
        self.q_ = np.stack([self.q_student_, self.q_emp_], axis=1).reshape(-1)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.q_ is None:
            raise RuntimeError("distribution model has not been fitted")
        return np.tile(self.q_, (x.shape[0], 1))

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="distribution",
            name="hstep",
            version="v1",
            extra={
                "horizons": list(self.horizons),
                "constructions": ["student_t", "empirical"],
                "column_layout": "per-horizon blocks: [student_t(taus) | empirical(taus)]",
                "n_columns": 2 * len(self.taus) * len(self.horizons),
                "mu_1": self.mu_,
                "sigma_1": self.sig_,
                "nu": self.nu_,
            },
        )
