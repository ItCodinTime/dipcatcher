"""WATCH: weighted-conformal test martingales for sequential monitoring.

Per-step weighted conformal p-values on a stream of nonconformity scores are
combined with a betting scheme into a test martingale M_t (M_0 = 1); an alarm
is raised when M_t >= 1/alpha, giving time-uniform false-alarm probability
<= alpha by Ville's inequality (Ville, 1939, *Etude critique de la notion de
collectif*; Ville, 1939, pp. 100-101 — P(sup_t M_t >= 1/alpha) <= alpha for a
nonnegative martingale started at 1). This is the weighted-conformal test
martingale (WCTM) of Prinster, Han & Saria (2025, "Adaptive Monitoring for AI
Deployments via Weighted-Conformal Martingales", arXiv:2505.04608, ICML):
online weighted-conformal p-values (their Eq. (9)) are exactly iid Unif[0, 1]
under the weighted null when the weights are correct (their Theorem 3.3), so
any predictable betting strategy that is a nonnegative martingale under iid
uniform p-values yields an anytime-valid monitor. With uniform weights this
reduces to the standard conformal test martingale of Vovk and coauthors
(Vovk, 2021, "Testing randomness online" + Vovk et al., 2022, arXiv:2202.13095,
"Conformal testing: binary versus regression" — conformal p-values from
Vovk, Gammerman & Shafer, 2005, *Algorithmic Learning in a Random World*,
Springer; the online pooled-score computation follows the online compression
model of Vovk, 2002/2003; first CTM betting schemes: Volkhonskiy et al., 2017,
COPA 2017, "Bernoulli-based tests for online exchangeability", arXiv:1706.02244;
Vovk et al., 2003, NIPS). The weighted p-value weights the score distribution
before ranking, in the sense of Tibshirani, Foygel Barber, Candes & Ramdas
(2019, Ann. Statist. 47(2):811-839, arXiv:1904.06001 — weighted conformal
prediction under covariate shift with likelihood-ratio weights); here the
weights come from a user-supplied ``weight_fn(i, t)`` (see
``quant_fund.models.weighted_conformal.likelihood_ratio_weights`` for a
histogram density-ratio estimator), letting the monitor either adapt to an
anticipated shift (weights tracking the shifted regime) or stay anchored to
the calibration regime (weights emphasizing the calibration past), per
Prinster et al. 2025, Sec. 3-4.

Betting schemes (Shafer, 2021, "Testing by Betting: A Strategy for Statistical
and Scientific Communication", J. R. Stat. Soc. A 184(2):407-431;
Shafer & Vovk, 2019, *Game-Theoretic Foundations for Probability and
Statistics*, Wiley):

- 'power': the power martingale M_t = prod_{s<=t} gamma * p_s^{gamma - 1} with
  ``power_gamma`` (default 0.5, Shafer's square-root martingale); E[gamma
  p^{gamma-1}] = 1 exactly for p ~ Unif[0, 1] and any gamma in (0, 1).
- 'mix' (default): an equal mixture over a fixed gamma grid
  (``DEFAULT_GAMMA_GRID`` = linspace(0.05, 0.95, 19)): component wealths
  evolve as independent power martingales and M_t is their mean, so M_t is
  again a nonnegative martingale with M_0 = 1 (a mixture of martingales);
  it removes the single-gamma tuning that governs the power martingale's
  detection delay (Shafer, 2021, Sec. 4.3, "betting strategies" / the
  plug-in and mixture constructions; cf. Grunwald, de Heide & Koolen, 2024,
  "Safe Testing", arXiv:1906.07801, Sec. 5, for mixture e-processes).

Honesty: outputs are p-values, martingale paths, alarm times, and detection
delays — proper sequential-testing quantities. No Sharpe/P&L metrics; no
live-trading claims (AGENTS.md honesty contract).

Conventions: numpy core, frozen dataclass result, fail-closed edges
(ValueError; non-finite or empty scores are rejected, never silently
dropped), seeded tie-breaking via ``np.random.default_rng``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.evalues import e_process_threshold

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    "DEFAULT_GAMMA_GRID",
    "WatchMartingale",
    "WatchResult",
    "conformity_from_quantiles",
    "run_watch",
]

_EPS = 1e-12
_E_MAX = 1e300
_MIN_BURN_IN = 5

# Equal-mixture grid for betting='mix'; gamma in (0, 1) as required by the
# power martingale (E[gamma p^{gamma-1}] = 1 fails at gamma = 0 and gamma = 1).
DEFAULT_GAMMA_GRID: Array = np.linspace(0.05, 0.95, 19)

WeightFn = Callable[[IntArray, int], Array]


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def _check_burn_in(burn_in: int) -> int:
    b = int(burn_in)
    if b < _MIN_BURN_IN:
        raise ValueError(f"burn_in must be >= {_MIN_BURN_IN}")
    return b


def _check_betting(betting: str) -> str:
    if betting not in ("power", "mix"):
        raise ValueError("betting must be 'power' or 'mix'")
    return betting


def _check_gamma(power_gamma: float) -> float:
    g = float(power_gamma)
    if not 0.0 < g < 1.0:
        raise ValueError("power_gamma must be in (0, 1)")
    return g


def _check_gamma_grid(gamma_grid: Array | None) -> Array:
    if gamma_grid is None:
        return DEFAULT_GAMMA_GRID
    grid = np.asarray(gamma_grid, dtype=float).reshape(-1)
    if grid.size < 2 or not np.all(np.isfinite(grid)) or not np.all((grid > 0.0) & (grid < 1.0)):
        raise ValueError("gamma_grid must hold >= 2 finite values in (0, 1)")
    return grid


def _check_scores(scores: Array) -> Array:
    s = np.asarray(scores, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("scores must be non-empty")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    return s


def conformity_from_quantiles(
    y: Array,
    quantile_grid: Array,
    quantile_levels: Array,
) -> Array:
    """Nonconformity score |y - q_0.5| / (q_0.75 - q_0.25) from a quantile grid.

    Exact score used, for each row t with forecast quantiles q_tau = grid[t]:

        s_t = |y_t - q_0.5,t| / max(q_0.75,t - q_0.25,t, 1e-12)

    i.e. the absolute residual from the interpolated median, scaled by the
    interpolated interquantile (IQR proxy) width so that unitless targets of
    different volatilities give comparable scores. q_tau is linear
    interpolation of ``quantile_levels`` -> ``quantile_grid`` row (np.interp
    semantics; if the grid does not cover 0.25/0.5/0.75 the nearest endpoint
    level is used, and the caller should treat the score as approximate). The
    scale is floored at 1e-12 only to avoid division by zero on degenerate
    grids; larger scores mean stranger points, as ``WatchMartingale`` expects.

    Parameters
    ----------
    y:
        Observations, shape (n,).
    quantile_grid:
        Forecast quantiles, shape (n, k), row t aligned with y_t.
    quantile_levels:
        Strictly increasing levels in (0, 1), shape (k,).

    Raises
    ------
    ValueError
        On shape mismatch, non-increasing levels, or non-finite inputs.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    grid = np.asarray(quantile_grid, dtype=float)
    levels = np.asarray(quantile_levels, dtype=float).reshape(-1)
    if y_arr.size == 0 or not np.all(np.isfinite(y_arr)):
        raise ValueError("y must be non-empty and finite")
    if grid.ndim != 2 or grid.shape[0] != y_arr.size or grid.shape[1] != levels.size:
        raise ValueError("quantile_grid must have shape (len(y), len(quantile_levels))")
    if levels.size < 2 or not np.all(np.isfinite(levels)):
        raise ValueError("quantile_levels must hold >= 2 finite values")
    if not np.all(np.diff(levels) > 0.0) or not np.all((levels > 0.0) & (levels < 1.0)):
        raise ValueError("quantile_levels must be strictly increasing in (0, 1)")
    if not np.all(np.isfinite(grid)):
        raise ValueError("quantile_grid must be finite")
    med = np.array([np.interp(0.5, levels, row) for row in grid], dtype=float)
    q75 = np.array([np.interp(0.75, levels, row) for row in grid], dtype=float)
    q25 = np.array([np.interp(0.25, levels, row) for row in grid], dtype=float)
    scale = np.maximum(q75 - q25, _EPS)
    return np.asarray(np.abs(y_arr - med) / scale, dtype=np.float64)


def _weights_at(weight_fn: WeightFn, t: int) -> Array:
    idx = np.arange(1, t + 1, dtype=np.int64)
    w = np.asarray(weight_fn(idx, t), dtype=float).reshape(-1)
    if w.shape != (t,):
        raise ValueError("weight_fn must return one weight per index")
    return w


def _weighted_p_value(history: Array, weights: Array, u: float) -> float:
    """Weighted-conformal p-value (Prinster et al. 2025, Eq. (9)).

    ``history`` holds scores s_1..s_t (current score last, larger = stranger),
    ``weights`` the unnormalized likelihood-ratio-style weights w_1..w_t.
    With w~_i = w_i / sum_j w~_j:

        p_t = sum_{i<t} w~_i 1{s_i > s_t} + w~_t * u

    where u ~ Unif[0, 1] breaks the self-tie. Uniform weights give the
    ordinary conformal p-value (Vovk et al., 2005); non-uniform weights give
    the weighted variant (Tibshirani et al., 2019, score-distribution
    weighting). Weights must be positive and finite (fail-closed).
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    s = np.asarray(history, dtype=float).reshape(-1)
    if w.shape != s.shape or s.size == 0:
        raise ValueError("weights must align with history")
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("weights must be positive and finite")
    w = w / w.sum()
    s_t = s[-1]
    strict = float(np.dot(w[:-1], (s[:-1] > s_t).astype(float)))
    return float(np.clip(strict + w[-1] * float(u), _EPS, 1.0))


def _uniform_p_values(scores: Array, u: Array) -> Array:
    """Ordinary online conformal p-values, vectorized over the whole stream.

    p_t = (sum_{i<=t} 1{s_i > s_t} + u_t) / t with the diagonal term
    contributing u_t (self-tie). Valid even for repeated scores because of
    the u_t term.
    """
    n = scores.size
    greater = np.triu(scores[:, None] > scores[None, :])
    counts = greater.sum(axis=0).astype(float)
    return np.clip((counts + u) / (np.arange(n, dtype=float) + 1.0), _EPS, 1.0)


def _weighted_p_values(scores: Array, weight_fn: WeightFn, rng: np.random.Generator) -> Array:
    n = scores.size
    p = np.empty(n, dtype=float)
    for t in range(1, n + 1):
        w = _weights_at(weight_fn, t)
        u = float(np.clip(rng.random(), _EPS, 1.0))
        p[t - 1] = _weighted_p_value(scores[:t], w, u)
    return p


def _power_log_factors(p_values: Array, gamma: float) -> Array:
    """log(gamma * p^{gamma-1}) per step, clipped to [_EPS, _E_MAX]."""
    pv = np.clip(p_values, _EPS, 1.0)
    return np.log(np.clip(gamma * pv ** (gamma - 1.0), _EPS, _E_MAX))


def _mixture_log_wealths(p_values: Array, gamma_grid: Array, burn_in: int) -> Array:
    """Per-component log-wealth, shape (n_steps, n_components).

    Burn-in steps contribute a unit factor (log-factor 0) BEFORE the
    cumulative product, so wealth stays 1 through burn-in and only post
    burn-in factors accumulate — matching the stateful class exactly.
    """
    pv = np.clip(p_values, _EPS, 1.0)[:, None]
    factors = np.clip(gamma_grid[None, :] * pv ** (gamma_grid[None, :] - 1.0), _EPS, _E_MAX)
    log_f = np.log(factors)
    log_f[:burn_in] = 0.0
    return np.minimum(np.cumsum(log_f, axis=0), np.log(_E_MAX))


def _martingale_path(
    p_values: Array,
    betting: str,
    power_gamma: float,
    gamma_grid: Array,
    burn_in: int,
) -> Array:
    """M_0..M_T with M_0 = 1 at index 0; no bet during burn-in (factor 1)."""
    n = p_values.size
    if betting == "power":
        log_f = _power_log_factors(p_values, power_gamma)
        log_f[:burn_in] = 0.0
        path = np.empty(n + 1, dtype=float)
        path[0] = 1.0
        path[1:] = np.exp(np.minimum(np.cumsum(log_f), np.log(_E_MAX)))
        return path
    log_w = _mixture_log_wealths(p_values, gamma_grid, burn_in)
    path = np.empty(n + 1, dtype=float)
    path[0] = 1.0
    path[1:] = np.minimum(np.exp(log_w).mean(axis=1), _E_MAX)
    return path


@dataclass(frozen=True)
class WatchResult:
    """Stateless ``run_watch`` outcome. ``martingale_path[t]`` = M_t (M_0 = 1)."""

    alarm_time: int | None
    martingale_path: Array
    p_values: Array


def run_watch(
    scores: Array,
    *,
    alpha: float = 0.05,
    weight_fn: WeightFn | None = None,
    betting: str = "mix",
    power_gamma: float | None = None,
    burn_in: int = 30,
    random_state: int | None = None,
    gamma_grid: Array | None = None,
) -> WatchResult:
    """Run a WATCH monitor over a complete score stream, statelessly.

    Parameters
    ----------
    scores:
        One-dimensional nonconformity scores s_1..s_T (larger = stranger),
        already computed by the caller (or via ``conformity_from_quantiles``).
        Must be non-empty and finite.
    alpha:
        Alarm level in (0, 1); alarm when M_t >= 1/alpha (Ville).
    weight_fn:
        Optional ``weight_fn(i, t) -> weights`` giving a positive weight for
        each stream index i (1-based, as a 1-D int array) at step t. None ->
        uniform weights (ordinary conformal p-values, Vovk et al. 2005).
        Non-uniform weights implement the Tibshirani et al. (2019) /
        Prinster et al. (2025) weighted-conformal p-value.
    betting:
        'power' (single-gamma power martingale, ``power_gamma``) or 'mix'
        (default; equal mixture over ``gamma_grid``, Shafer 2021).
    power_gamma:
        Power-martingale parameter in (0, 1); default 0.5 (square-root
        martingale). Ignored unless betting='power'.
    burn_in:
        Steps with no bet (factor 1) while the score pool is too small to
        rank meaningfully; must be >= 5.
    random_state:
        Seed for the tie-breaking uniforms u_t.
    gamma_grid:
        Mixture grid (all values in (0, 1)); default ``DEFAULT_GAMMA_GRID``.

    Returns
    -------
    WatchResult
        ``alarm_time`` is the first step t with M_t >= 1/alpha (None if no
        alarm), ``martingale_path`` is M_0..M_T (index 0 = M_0 = 1), and
        ``p_values`` the per-step weighted-conformal p-values.
    """
    a = _check_alpha(alpha)
    b = _check_burn_in(burn_in)
    kind = _check_betting(betting)
    gamma = _check_gamma(0.5 if power_gamma is None else power_gamma)
    grid = _check_gamma_grid(gamma_grid)
    s = _check_scores(scores)
    rng = np.random.default_rng(random_state)
    if weight_fn is None:
        u = np.clip(rng.random(s.size), _EPS, 1.0)
        p_values = _uniform_p_values(s, u)
    else:
        p_values = _weighted_p_values(s, weight_fn, rng)
    path = _martingale_path(p_values, kind, gamma, grid, b)
    decision = e_process_threshold(path, level=a)
    first = cast("int | None", decision["first_cross"])
    alarm = None if first is None else int(first)
    return WatchResult(alarm_time=alarm, martingale_path=path, p_values=p_values)


class WatchMartingale:
    """Stateful WATCH monitor; feed one nonconformity score at a time.

    Parameters as in ``run_watch``; ``update(score)`` appends one score,
    computes the weighted-conformal p-value against the pooled history,
    multiplies the betting wealth (no bet during burn-in), and returns the
    current M_t. ``alarmed`` is True once M_t >= 1/alpha; streaming continues
    after an alarm so callers can inspect the path. The p-value pool is the
    full stream observed so far (online compression model, Vovk 2002/2003),
    matching ``run_watch`` exactly: both produce identical paths given the
    same seed and weight function.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        weight_fn: WeightFn | None = None,
        betting: str = "mix",
        power_gamma: float | None = None,
        burn_in: int = 30,
        random_state: int | None = None,
        gamma_grid: Array | None = None,
    ) -> None:
        self.alpha = _check_alpha(alpha)
        self.burn_in = _check_burn_in(burn_in)
        self.betting = _check_betting(betting)
        self.power_gamma = _check_gamma(0.5 if power_gamma is None else power_gamma)
        self.gamma_grid = _check_gamma_grid(gamma_grid)
        self.weight_fn = weight_fn
        self._rng = np.random.default_rng(random_state)
        self._scores: list[float] = []
        self._p_values: list[float] = []
        self._wealths = np.ones(self.gamma_grid.size, dtype=float)
        self._m = 1.0
        self._path: list[float] = [1.0]

    def update(self, score: float) -> float:
        """Append one nonconformity score (larger = stranger); return M_t."""
        s_t = float(score)
        if not np.isfinite(s_t):
            raise ValueError("score must be finite")
        self._scores.append(s_t)
        t = len(self._scores)
        history = np.asarray(self._scores, dtype=float)
        w = np.ones(t, dtype=float) if self.weight_fn is None else _weights_at(self.weight_fn, t)
        u = float(np.clip(self._rng.random(), _EPS, 1.0))
        p = _weighted_p_value(history, w, u)
        self._p_values.append(p)
        if t > self.burn_in:
            pv = np.clip(p, _EPS, 1.0)
            if self.betting == "power":
                factor = float(
                    np.clip(self.power_gamma * pv ** (self.power_gamma - 1.0), _EPS, _E_MAX)
                )
                self._m = min(self._m * factor, _E_MAX)
            else:
                factors = np.clip(self.gamma_grid * pv ** (self.gamma_grid - 1.0), _EPS, _E_MAX)
                self._wealths = self._wealths * factors
                self._m = float(min(self._wealths.mean(), _E_MAX))
        self._path.append(self._m)
        return self._m

    @property
    def alarmed(self) -> bool:
        """True when M_t >= 1/alpha (time-uniform FPR <= alpha by Ville)."""
        return self._m >= 1.0 / self.alpha

    @property
    def p_value_history(self) -> Array:
        """Per-step weighted-conformal p-values observed so far."""
        return np.asarray(self._p_values, dtype=float)

    @property
    def martingale_path(self) -> Array:
        """M_0..M_t with M_0 = 1 at index 0."""
        return np.asarray(self._path, dtype=float)

    @property
    def step(self) -> int:
        """Number of scores consumed so far (t)."""
        return len(self._scores)
