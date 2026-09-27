"""WATCH: weighted conformal test martingales for shift monitoring.

Prinster, Han, Liu & Saria (2025), "WATCH: Adaptive Monitoring for AI
Deployments via Weighted-Conformal Martingales", ICML, PMLR 267
(arXiv:2505.04608). A conformal test martingale (Vovk, Gammerman & Shafer
2005; Vovk 2021) multiplies a betting function of online conformal p-values.
Under exchangeability those p-values are uniform, the product is a test
martingale, and Ville's inequality bounds the chance it ever reaches 1/α.

WATCH replaces the uniform weights in the p-value with density-ratio weights
once a parallel covariate martingale (the X-CTM) crosses an adaptation
threshold, so a correctly specified covariate shift does not by itself raise
the label martingale (their Theorem 3.1 and Eq. 17). Root-cause labels follow
Section 4.4: label alarm without an X alarm is a concept shift; both alarms
are an extreme covariate shift; adaptation without a label alarm is a benign
covariate shift.

The betting capital is the simple jumper of Vovk (2021), arXiv:2105.08669
Algorithm 1, averaged over a grid of jump rates that includes J=1 (the mean
jumper: one component is identically 1, so the average never falls below
1/|J|). Ties in the p-value use the conservative indicator 1{v_i ≥ v_test}
(u=1 in their Eq. 9), which keeps the capital a supermartingale. The density
ratio defaults to a diagonal-Gaussian plugin on the calibration versus
post-adaptation covariates. That plugin is not the paper's neural classifier;
pass ``weight_fn`` to supply oracle or estimated weights. No Sharpe / P&L.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "WATCHMonitor",
    "WATCHStep",
    "CompositeJumper",
    "SimpleJumper",
    "diagnose_shift",
    "gaussian_density_ratio",
    "nearest_neighbor_scores",
    "shiryaev_roberts",
    "weighted_conformal_pvalue",
]

_EPSILONS: tuple[float, float, float] = (-1.0, 0.0, 1.0)
_E_MAX = 1e300
_LOG_RATIO_CAP = 20.0
WeightFn = Callable[[Array, Array], tuple[Array, Array]]


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def betting_factor(p_value: float, epsilon: float) -> float:
    """h_ε(p) = 1 + ε (p − 0.5), ε ∈ {−1, 0, 1} (Vovk 2021, Eq. 1)."""
    return 1.0 + float(epsilon) * (float(p_value) - 0.5)


class SimpleJumper:
    """Simple jumper betting martingale (Vovk 2021, arXiv:2105.08669, Algorithm 1)."""

    def __init__(self, jump_rate: float = 0.01) -> None:
        rate = float(jump_rate)
        if not np.isfinite(rate) or not 0.0 < rate <= 1.0:
            raise ValueError("jump_rate must be in (0, 1]")
        self.jump_rate = rate
        self.reset()

    def reset(self) -> None:
        """Restore equal capital on ε ∈ {−1, 0, 1} and wealth 1."""
        self._capital = np.full(3, 1.0 / 3.0, dtype=np.float64)
        self.value = 1.0
        self.n_seen = 0

    def update(self, p_value: float) -> float:
        """Consume one p-value and return the jumper wealth S_n."""
        u = float(p_value)
        if not np.isfinite(u) or u < 0.0 or u > 1.0:
            raise ValueError("p-value must be in [0, 1]")
        total = float(self._capital.sum())
        rate = self.jump_rate
        self._capital = (1.0 - rate) * self._capital + (rate / 3.0) * total
        factors = np.array([betting_factor(u, eps) for eps in _EPSILONS], dtype=np.float64)
        self._capital = self._capital * factors
        wealth = float(self._capital.sum())
        if wealth > _E_MAX:
            self._capital *= _E_MAX / wealth
            wealth = _E_MAX
        self.value = wealth
        self.n_seen += 1
        return self.value


class CompositeJumper:
    """Mean of simple jumpers. The J=1 component is identically 1 (wealth floor)."""

    def __init__(self, jump_rates: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)) -> None:
        if len(jump_rates) == 0:
            raise ValueError("jump_rates must be non-empty")
        self.jumpers = [SimpleJumper(rate) for rate in jump_rates]

    def reset(self) -> None:
        for jumper in self.jumpers:
            jumper.reset()

    def update(self, p_value: float) -> float:
        values = [jumper.update(p_value) for jumper in self.jumpers]
        return float(np.mean(np.asarray(values, dtype=float)))

    @property
    def value(self) -> float:
        return float(np.mean([jumper.value for jumper in self.jumpers]))

    @property
    def floor(self) -> float:
        """Lower bound 1/|J| when some jump rate is 1; otherwise 0."""
        if any(jumper.jump_rate == 1.0 for jumper in self.jumpers):
            return 1.0 / len(self.jumpers)
        return 0.0


def weighted_conformal_pvalue(
    scores: Array,
    weights: Array | None = None,
    test_index: int = -1,
) -> float:
    """Conservative weighted conformal p-value (Prinster et al. 2025, Eq. 9, u=1).

    p = Σ_i w̃_i 1{v_i ≥ v_test} with w̃ summing to 1. Uniform weights recover
    the rank p-value. The test score must be one of ``scores``.
    """
    vals = np.asarray(scores, dtype=float).reshape(-1)
    if vals.size == 0 or not np.all(np.isfinite(vals)):
        raise ValueError("scores must be non-empty and finite")
    idx = int(test_index)
    if idx < 0:
        idx = vals.size + idx
    if idx < 0 or idx >= vals.size:
        raise ValueError("test_index is out of range")
    if weights is None:
        raw = np.ones(vals.size, dtype=np.float64)
    else:
        raw = np.asarray(weights, dtype=float).reshape(-1)
        if raw.shape != vals.shape:
            raise ValueError("weights must match scores")
        if not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
            raise ValueError("weights must be finite and nonnegative")
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have positive finite sum")
    test = float(vals[idx])
    return float(raw[vals >= test].sum() / total)


def nearest_neighbor_scores(x: Array) -> Array:
    """Leave-one-out nearest-neighbor distance, the X nonconformity of Vovk et al. (2021).

    ``x`` is ``(n,)`` or ``(n, d)`` with n ≥ 2. Score i is the Euclidean distance
    from row i to the nearest other row.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError("x must contain at least two finite rows")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x must be finite")
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    scores = np.min(dist, axis=1)
    if not np.all(np.isfinite(scores)):
        raise ValueError("nearest-neighbor scores are non-finite")
    return np.asarray(scores, dtype=np.float64)


def gaussian_density_ratio(x_calibration: Array, x_test: Array) -> tuple[Array, Array]:
    """Diagonal-Gaussian plugin for g(x)/f(x).

    Source moments come from ``x_calibration``. Target moments come from
    ``x_test`` when it has at least two rows; a single test row uses the
    source variance and that row's mean (a one-point mean-shift plugin).
    Log ratios are clipped to ±20. Feature shape is ``(n, d)``.
    """
    cal = np.asarray(x_calibration, dtype=float)
    test = np.asarray(x_test, dtype=float)
    if cal.ndim == 1:
        cal = cal.reshape(-1, 1)
    if test.ndim == 1:
        test = test.reshape(-1, 1)
    if cal.ndim != 2 or test.ndim != 2 or cal.shape[1] != test.shape[1]:
        raise ValueError("calibration and test covariates must share feature dimension")
    if cal.shape[0] < 2 or test.shape[0] < 1:
        raise ValueError("gaussian plugin needs at least two calibration rows and one test row")
    if not np.all(np.isfinite(cal)) or not np.all(np.isfinite(test)):
        raise ValueError("covariates must be finite")
    mu_s = np.mean(cal, axis=0)
    var_s = np.var(cal, axis=0, ddof=1)
    if np.any(var_s <= 0.0) or not np.all(np.isfinite(var_s)):
        raise ValueError("calibration covariate variance is degenerate")
    if test.shape[0] >= 2:
        mu_t = np.mean(test, axis=0)
        var_t = np.var(test, axis=0, ddof=1)
        if np.any(var_t <= 0.0) or not np.all(np.isfinite(var_t)):
            var_t = var_s.copy()
    else:
        mu_t = test[0].copy()
        var_t = var_s.copy()

    def _weights(rows: Array) -> Array:
        z_s = (rows - mu_s) ** 2 / var_s
        z_t = (rows - mu_t) ** 2 / var_t
        log_ratio = -0.5 * np.sum(np.log(var_t / var_s) + z_t - z_s, axis=-1)
        log_ratio = np.clip(log_ratio, -_LOG_RATIO_CAP, _LOG_RATIO_CAP)
        return np.asarray(np.exp(log_ratio), dtype=np.float64)

    return _weights(cal), _weights(test)


def diagnose_shift(y_alarm: bool, x_alarm: bool, adapted: bool) -> str:
    """Section 4.4 root cause from sticky alarm flags.

    ``extreme_covariate`` if both martingales have alarmed, ``concept`` if only
    the label martingale has, ``benign_covariate`` if the monitor has adapted
    and the label martingale has not, otherwise ``none``.
    """
    if y_alarm and x_alarm:
        return "extreme_covariate"
    if y_alarm:
        return "concept"
    if adapted:
        return "benign_covariate"
    return "none"


def shiryaev_roberts(wealth: Array) -> float:
    """Shiryaev–Roberts statistic Σ_{i<t} M_t / M_i (Prinster et al. 2025, Eq. 15).

    ``wealth`` includes M_0. The statistic uses every entry except the last as
    a past value. Requires strictly positive finite wealth.
    """
    path = np.asarray(wealth, dtype=float).reshape(-1)
    if path.size < 2 or not np.all(np.isfinite(path)) or np.any(path <= 0.0):
        raise ValueError("wealth path must contain at least M_0 and one update, all positive")
    current = float(path[-1])
    return float(current * np.sum(1.0 / path[:-1]))


@dataclass(frozen=True)
class WATCHStep:
    """One online WATCH update."""

    p_y: float
    p_x: float
    martingale_y: float
    martingale_x: float
    shiryaev_roberts_y: float
    adapted: bool
    alarm: bool
    x_alarm: bool
    diagnosis: str
    out_of_support: bool


class WATCHMonitor:
    """Online WATCH monitor on a label nonconformity score and a covariate vector.

    ``update(y_score, x)`` appends one observation. Before adaptation the label
    p-value is the uniform rank on the growing score bag. When the X-CTM
    reaches ``adapt_threshold`` and at least ``min_calibration`` label scores
    are already stored, those scores and covariates freeze as the calibration
    bag (Eq. 17). Later label p-values use ``weight_fn(x_cal, x_test)`` or the
    Gaussian plugin. Alarms are sticky: a martingale that later falls does not
    clear Ville's crossing. ``adapt_threshold`` defaults to ``1/sqrt(α)``, a
    lab default below the Ville level ``1/α``, not a constant from the paper.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        adapt_threshold: float | None = None,
        jump_rates: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0),
        weight_fn: WeightFn | None = None,
        min_calibration: int = 20,
        weight_cap: float = 1e6,
    ) -> None:
        self.alpha = _check_alpha(alpha)
        if adapt_threshold is None:
            threshold = 1.0 / np.sqrt(self.alpha)
        else:
            threshold = float(adapt_threshold)
        if not np.isfinite(threshold) or threshold <= 1.0:
            raise ValueError("adapt_threshold must be finite and > 1")
        cal = int(min_calibration)
        if cal < 2:
            raise ValueError("min_calibration must be at least 2")
        cap = float(weight_cap)
        if not np.isfinite(cap) or cap <= 1.0:
            raise ValueError("weight_cap must be finite and > 1")
        self.adapt_threshold = float(threshold)
        self.alarm_level = 1.0 / self.alpha
        self.min_calibration = cal
        self.weight_cap = cap
        self.weight_fn = weight_fn
        self._y_jumper = CompositeJumper(jump_rates)
        self._x_jumper = CompositeJumper(jump_rates)
        self._y: list[float] = []
        self._x: list[Array] = []
        self._test_x: list[Array] = []
        self._cal_y: Array | None = None
        self._cal_x: Array | None = None
        self._adapted = False
        self._y_alarm = False
        self._x_alarm = False
        self._wealth_y: list[float] = [1.0]
        self._x_dim: int | None = None

    @property
    def adapted(self) -> bool:
        return self._adapted

    @property
    def alarm(self) -> bool:
        return self._y_alarm

    def update(self, y_score: float, x: Array | float) -> WATCHStep:
        """Incorporate one label score and its covariate. Returns the step record."""
        score = float(y_score)
        if not np.isfinite(score):
            raise ValueError("y_score must be finite")
        features = np.asarray(x, dtype=float).reshape(-1)
        if features.size == 0 or not np.all(np.isfinite(features)):
            raise ValueError("x must be a non-empty finite covariate")
        if self._x_dim is None:
            self._x_dim = int(features.size)
        elif features.size != self._x_dim:
            raise ValueError("x feature dimension changed")
        self._x.append(features.copy())
        p_x = self._x_pvalue()
        m_x = self._x_jumper.update(p_x)
        if m_x >= self.alarm_level:
            self._x_alarm = True
        self._maybe_adapt(m_x)
        p_y, out_of_support = self._y_pvalue(score)
        m_y = self._y_jumper.update(p_y)
        self._wealth_y.append(m_y)
        if m_y >= self.alarm_level:
            self._y_alarm = True
        return WATCHStep(
            p_y=p_y,
            p_x=p_x,
            martingale_y=m_y,
            martingale_x=m_x,
            shiryaev_roberts_y=shiryaev_roberts(np.asarray(self._wealth_y, dtype=float)),
            adapted=self._adapted,
            alarm=self._y_alarm,
            x_alarm=self._x_alarm,
            diagnosis=diagnose_shift(self._y_alarm, self._x_alarm, self._adapted),
            out_of_support=out_of_support,
        )

    def _x_pvalue(self) -> float:
        if len(self._x) == 1:
            return 1.0
        mat = np.stack(self._x, axis=0)
        return weighted_conformal_pvalue(nearest_neighbor_scores(mat))

    def _maybe_adapt(self, m_x: float) -> None:
        if self._adapted:
            return
        if m_x < self.adapt_threshold:
            return
        if len(self._y) < self.min_calibration:
            return
        self._cal_y = np.asarray(self._y, dtype=np.float64)
        self._cal_x = np.stack(self._x[:-1], axis=0)
        self._adapted = True

    def _y_pvalue(self, score: float) -> tuple[float, bool]:
        if not self._adapted:
            self._y.append(score)
            return weighted_conformal_pvalue(np.asarray(self._y, dtype=float)), False
        assert self._cal_y is not None and self._cal_x is not None
        self._test_x.append(self._x[-1].copy())
        x_test = np.stack(self._test_x, axis=0)
        if self.weight_fn is None:
            w_cal, w_test = gaussian_density_ratio(self._cal_x, x_test)
        else:
            w_cal, w_test = self.weight_fn(self._cal_x, x_test)
            w_cal = np.asarray(w_cal, dtype=float).reshape(-1)
            w_test = np.asarray(w_test, dtype=float).reshape(-1)
        if w_cal.shape != self._cal_y.shape or w_test.shape[0] != x_test.shape[0]:
            raise ValueError("weight_fn returned weights of the wrong shape")
        if not np.all(np.isfinite(w_cal)) or not np.all(np.isfinite(w_test)):
            raise ValueError("weight_fn returned non-finite weights")
        if np.any(w_cal < 0.0) or np.any(w_test < 0.0):
            raise ValueError("weight_fn returned negative weights")
        out_of_support = bool(np.max(w_test) > self.weight_cap or np.max(w_cal) > self.weight_cap)
        scores = np.concatenate([self._cal_y, np.asarray([score], dtype=float)])
        weights = np.concatenate([w_cal, np.asarray([w_test[-1]], dtype=float)])
        return weighted_conformal_pvalue(scores, weights), out_of_support
