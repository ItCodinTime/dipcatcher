"""Fleet evaluation of distribution challengers on seeded SYNTHETIC shards.

Each shard generator draws a labeled synthetic return series with a planted
distributional property (multimodality, heavy tails, skew, regime-switching,
volatility breaks, GJR-GARCH leverage asymmetry, or a causal lag-1 feature
frame). ``run_distribution_fleet`` fits every head on the leading slice of
every shard, predicts the trailing slice, and scores with proper scores only
(pinball, CRPS, PIT-KS, interval coverage) — never headline P&L metrics.
``write_fleet_receipt`` persists the sealed JSON evidence under ``receipts/``.

Conditional/series heads enter through fleet adapters: ``_QarOneStepHead``
scores QAR(1) on its native one-step grid (each eval row forecast from the
*observed* lag-1 value — frozen coefficients, no refit, no lookahead), and
``_HStepOneStepHead`` slices the ``h=1`` block of the multi-horizon head so
only its honestly one-step forecasts are scored. All output is SYNTHETIC
correctness evidence, not market data.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.probability import pit_ks
from quant_fund.metrics.scoring import (
    coverage,
    crps_from_quantiles,
    mean_pinball,
    pit_values,
    rearrange_quantiles,
)
from quant_fund.models.conformal_dist import ConformalTDistribution
from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
    GMMDistribution,
    IsotonicPitDistribution,
    SkewTDistribution,
    StackedDistribution,
)
from quant_fund.models.fhs import FhsSkewDistribution
from quant_fund.models.hstep import HStepScaledDistribution
from quant_fund.models.lgbm_q2 import LGBMQ2Distribution
from quant_fund.models.qar import QARDistribution
from quant_fund.models.regime_dist import RegimeDistribution
from quant_fund.models.skew_t import skew_t_ppf
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

FLEET_EVAL_SCHEMA = "fleet_eval.v1"
DEFAULT_TAUS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
COVERAGE_LEVELS: tuple[float, ...] = (0.8, 0.9)


@dataclass(frozen=True)
class SyntheticShard:
    """One seeded SYNTHETIC return shard: design matrix + target series.

    Most shards carry the inert ``_dummy_x`` (unconditional heads ignore it);
    shards like ``ar1_lagged_x`` instead carry a causal feature frame.
    """

    name: str
    x: Array
    y: Array
    config: dict[str, Any]


ShardGenerator = Callable[[int, int], SyntheticShard]
HeadFactory = Callable[[], Any]


def _require_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("shard size n must be a positive integer")
    return int(n)


def _dummy_x(n: int) -> Array:
    """Inert single-column design matrix — unconditional heads ignore x."""
    return np.ones((_require_n(n), 1), dtype=np.float64)


def iid_gaussian(n: int, seed: int) -> SyntheticShard:
    """Homoskedastic N(mu, sigma) baseline shard (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "serial_dependence": False,
        "mu": 0.0,
        "sigma": 0.02,
    }
    y = rng.normal(config["mu"], config["sigma"], _require_n(n))
    return SyntheticShard("iid_gaussian", _dummy_x(n), np.asarray(y, dtype=float), config)


def bimodal_mixture(n: int, seed: int) -> SyntheticShard:
    """Two-component Gaussian mixture — planted multimodality (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "serial_dependence": False,
        "weight_lo": 0.65,
        "mu": (-0.025, 0.03),
        "sigma": (0.012, 0.016),
    }
    n = _require_n(n)
    hi = rng.random(n) >= float(config["weight_lo"])
    mus = np.asarray(config["mu"], dtype=float)
    sigs = np.asarray(config["sigma"], dtype=float)
    y = rng.normal(np.where(hi, mus[1], mus[0]), np.where(hi, sigs[1], sigs[0]))
    return SyntheticShard("bimodal_mixture", _dummy_x(n), np.asarray(y, dtype=float), config)


def heavy_tail(n: int, seed: int) -> SyntheticShard:
    """Student-t df=3 shard — planted heavy tails, infinite kurtosis (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "serial_dependence": False,
        "df": 3.0,
        "scale": 0.01,
    }
    y = rng.standard_t(float(config["df"]), _require_n(n)) * float(config["scale"])
    return SyntheticShard("heavy_tail", _dummy_x(n), np.asarray(y, dtype=float), config)


def left_skew(n: int, seed: int) -> SyntheticShard:
    """Hansen (1994) skew-t shard with lam < 0 — planted left skew (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "serial_dependence": False,
        "nu": 6.0,
        "lam": -0.6,
        "mu": 0.0,
        "sigma": 0.02,
    }
    u = np.clip(rng.uniform(0.0, 1.0, _require_n(n)), 1e-9, 1.0 - 1e-9)
    nu = float(config["nu"])
    lam = float(config["lam"])
    mu = float(config["mu"])
    sigma = float(config["sigma"])
    y = np.asarray([skew_t_ppf(float(uu), nu, lam, mu, sigma) for uu in u], dtype=np.float64)
    return SyntheticShard("left_skew", _dummy_x(n), y, config)


def regime_switch(n: int, seed: int) -> SyntheticShard:
    """Two-state Markov-switching vol AR(1) mixture (Hamilton 1989; SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "serial_dependence": True,
        "rho": 0.1,
        "sigma": (0.008, 0.035),
        "p_leave": (0.04, 0.08),
    }
    n = _require_n(n)
    sig_lo, sig_hi = config["sigma"]
    p01, p10 = config["p_leave"]
    state = np.empty(n, dtype=np.int64)
    state[0] = 0
    u = rng.random(n)
    for t in range(1, n):
        stay = (1.0 - p01) if state[t - 1] == 0 else (1.0 - p10)
        state[t] = state[t - 1] if u[t] < stay else 1 - state[t - 1]
    eps = rng.normal(0.0, np.where(state == 0, sig_lo, sig_hi))
    rho = float(config["rho"])
    y = np.empty(n, dtype=float)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = rho * y[t - 1] + eps[t]
    config["n_state1"] = int(state.sum())
    return SyntheticShard("regime_switch", _dummy_x(n), y, config)


def garch_cluster(n: int, seed: int) -> SyntheticShard:
    """GJR-GARCH(1,1) path (Glosten-Jagannathan-Runkle 1993; SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "serial_dependence": True,
        "omega": 4e-6,
        "alpha": 0.05,
        "gamma": 0.08,
        "beta": 0.9,
        "burn": 128,
    }
    n = _require_n(n)
    omega = float(config["omega"])
    alpha = float(config["alpha"])
    gamma = float(config["gamma"])
    beta = float(config["beta"])
    persistence = alpha + gamma / 2.0 + beta
    if not 0.0 < persistence < 1.0:
        raise ValueError("GJR-GARCH persistence must lie in (0, 1)")
    burn = int(config["burn"])
    total = n + burn
    z = rng.normal(0.0, 1.0, total)
    eps = np.empty(total, dtype=float)
    var = np.empty(total, dtype=float)
    var[0] = omega / (1.0 - persistence)
    eps[0] = np.sqrt(var[0]) * z[0]
    for t in range(1, total):
        var[t] = omega + (alpha + gamma * (eps[t - 1] < 0.0)) * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(var[t]) * z[t]
    config["persistence"] = float(persistence)
    return SyntheticShard("garch_cluster", _dummy_x(n), np.asarray(eps[burn:], dtype=float), config)


def vol_break(n: int, seed: int) -> SyntheticShard:
    """Single structural volatility break, low -> high sigma (SYNTHETIC).

    The break lands inside the leading fit window so both vol states are
    learnable, while the trailing slice is entirely high-vol: unconditional
    heads must over-mix the low-vol history, and regime-aware heads earn their
    score only if the filtered state tracks the break. Planted break for the
    ``regime`` cell.
    """
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "serial_dependence": True,
        "mu": 0.0,
        "sigma": (0.008, 0.04),
        "break_frac": 0.55,
    }
    n = _require_n(n)
    frac = float(config["break_frac"])
    if not 0.0 < frac < 1.0:
        raise ValueError("break_frac must lie in (0, 1)")
    cut = min(max(int(np.floor(frac * n)), 1), n - 1)
    sig_lo, sig_hi = config["sigma"]
    sigma = np.full(n, float(sig_lo))
    sigma[cut:] = float(sig_hi)
    y = rng.normal(float(config["mu"]), sigma)
    config["n_post_break"] = int(n - cut)
    return SyntheticShard("vol_break", _dummy_x(n), np.asarray(y, dtype=float), config)


def gjr_leverage(n: int, seed: int) -> SyntheticShard:
    """GJR-GARCH(1,1) with strong leverage + skew-t innovations (SYNTHETIC).

    Strong GJR gamma (negative shocks inflate next-step variance) over Hansen
    skew-t innovations (lam < 0) — the asymmetry ``fhs_skew``'s skew-t residual
    law is built to capture, unlike the symmetric-normal ``garch_cluster``.
    """
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "serial_dependence": True,
        "omega": 4e-6,
        "alpha": 0.03,
        "gamma": 0.18,
        "beta": 0.86,
        "nu": 8.0,
        "lam": -0.5,
        "burn": 128,
    }
    n = _require_n(n)
    omega = float(config["omega"])
    alpha = float(config["alpha"])
    gamma = float(config["gamma"])
    beta = float(config["beta"])
    persistence = alpha + gamma / 2.0 + beta
    if not 0.0 < persistence < 1.0:
        raise ValueError("GJR-GARCH persistence must lie in (0, 1)")
    burn = int(config["burn"])
    total = n + burn
    u = np.clip(rng.uniform(0.0, 1.0, total), 1e-9, 1.0 - 1e-9)
    nu = float(config["nu"])
    lam = float(config["lam"])
    z = np.asarray([skew_t_ppf(float(uu), nu, lam, 0.0, 1.0) for uu in u], dtype=np.float64)
    z_std = float(z.std(ddof=1))
    if not np.isfinite(z_std) or z_std <= 0.0:
        raise ValueError("skew-t innovation draw is degenerate")
    z = (z - float(z.mean())) / z_std
    eps = np.empty(total, dtype=float)
    var = np.empty(total, dtype=float)
    var[0] = omega / (1.0 - persistence)
    eps[0] = np.sqrt(var[0]) * z[0]
    for t in range(1, total):
        var[t] = omega + (alpha + gamma * (eps[t - 1] < 0.0)) * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(var[t]) * z[t]
    config["persistence"] = float(persistence)
    return SyntheticShard("gjr_leverage", _dummy_x(n), np.asarray(eps[burn:], dtype=float), config)


def ar1_lagged_x(n: int, seed: int) -> SyntheticShard:
    """AR(1) shard with a causal lag-1 feature frame (SYNTHETIC).

    ``y_t = rho * y_{t-1} + sigma * eps_t``; ``x`` carries ``[y_{t-1},
    |y_{t-1}|]`` — features known at the previous origin, so row ``t``'s
    features never contain its own target. Exercises ``lgbm_q2`` on a real
    signal and ``qar`` on a process whose conditional quantile map is linear
    in the lag by construction.
    """
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "serial_dependence": True,
        "rho": 0.35,
        "sigma": 0.02,
        "burn": 64,
        "x_features": ["y_lag1", "abs_y_lag1"],
    }
    n = _require_n(n)
    rho = float(config["rho"])
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie in (-1, 1)")
    sigma = float(config["sigma"])
    burn = int(config["burn"])
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    y_full = np.empty(total, dtype=float)
    y_full[0] = eps[0]
    for t in range(1, total):
        y_full[t] = rho * y_full[t - 1] + eps[t]
    y = np.asarray(y_full[burn:], dtype=float)
    lag = y_full[burn - 1 : -1]
    x = np.column_stack([lag, np.abs(lag)]).astype(np.float64)
    return SyntheticShard("ar1_lagged_x", x, y, config)


SHARD_GENERATORS: dict[str, ShardGenerator] = {
    "iid_gaussian": iid_gaussian,
    "bimodal_mixture": bimodal_mixture,
    "heavy_tail": heavy_tail,
    "left_skew": left_skew,
    "regime_switch": regime_switch,
    "garch_cluster": garch_cluster,
    "vol_break": vol_break,
    "gjr_leverage": gjr_leverage,
    "ar1_lagged_x": ar1_lagged_x,
}


class _QarOneStepHead:
    """Fleet adapter scoring ``QARDistribution`` on its native one-step grid.

    The head's ``predict`` emits exactly one row — the conditional quantile
    map ``a_tau + b_tau * y_{t-1}`` at the last fitted observation. For the
    trailing slice, eval row ``i`` is forecast with conditioning lag
    ``y[n_train + i - 1]`` (the observed previous value): the coefficient map
    stays frozen from the fit window, so this is a genuine one-step-ahead
    walk-forward score with no refit and no lookahead. The fleet passes the
    lag-1 column of ``y`` as the predict matrix (see ``fleet_lagged_predict``
    in ``_score_row``).
    """

    fleet_lagged_predict = True

    def __init__(self, taus: Sequence[float]) -> None:
        self.taus = [float(t) for t in taus]
        self._inner = QARDistribution(list(self.taus))

    def fit(self, x: Array, y: Array, **kwargs: Any) -> _QarOneStepHead:
        self._inner.fit(x, y)
        return self

    def predict(self, x: Array) -> Array:
        if self._inner.coef_ is None:
            raise RuntimeError("distribution model has not been fitted")
        lag = np.asarray(x, dtype=float).reshape(-1)
        if lag.size == 0 or not np.isfinite(lag).all():
            raise ValueError("lagged predictor column must be nonempty and finite")
        coef = self._inner.coef_
        q = coef[:, 0][None, :] + coef[:, 1][None, :] * lag[:, None]
        return np.asarray(rearrange_quantiles(q), dtype=np.float64)

    def metadata(self) -> Any:
        return self._inner.metadata()


class _HStepOneStepHead:
    """Scores one ``h=1`` construction block of ``HStepScaledDistribution``.

    The head's native ``predict`` emits ``2 * len(taus) * n_horizons`` columns
    (per-horizon ``[student_t | empirical]`` blocks). The fleet fits the
    default multi-horizon layout and scores only the ``h=1`` block — the
    one-step forecasts the trailing slice can honestly verify. The longer-
    horizon blocks are still emitted by the fit but are not scored here.
    """

    def __init__(self, taus: Sequence[float], block: str) -> None:
        if block not in {"student_t", "empirical"}:
            raise ValueError("block must be 'student_t' or 'empirical'")
        self.taus = [float(t) for t in taus]
        self.block = block
        self._inner = HStepScaledDistribution(list(self.taus))

    def fit(self, x: Array, y: Array, **kwargs: Any) -> _HStepOneStepHead:
        self._inner.fit(x, y)
        return self

    def predict(self, x: Array) -> Array:
        n_taus = len(self.taus)
        full = np.asarray(self._inner.predict(x), dtype=float)
        h1 = self._inner.horizons.index(1)
        lo = (2 * h1 + (0 if self.block == "student_t" else 1)) * n_taus
        return np.asarray(full[:, lo : lo + n_taus], dtype=np.float64)

    def metadata(self) -> Any:
        meta = self._inner.metadata()
        meta.extra = {**dict(meta.extra or {}), "fleet_block": self.block, "fleet_h": 1}
        return meta


# Head-name registry for the `fleet` CLI --models flag. Constructors take the
# scoring tau grid and a seed. Covers the unconditional heads from
# models/distribution.py plus the landed conditional/series heads via the
# fleet adapters above: qar (one-step lagged scoring), hstep as its two h=1
# construction slices, and the series/feature heads regime / fhs_skew /
# lgbm_q2 / conf_t directly.
FLEET_HEAD_REGISTRY: dict[str, Callable[[Sequence[float], int], Any]] = {
    "empirical": lambda taus, seed: EmpiricalDistribution(list(taus)),
    "gaussian": lambda taus, seed: GaussianDistribution(list(taus)),
    "skew_t": lambda taus, seed: SkewTDistribution(list(taus)),
    "gmm": lambda taus, seed: GMMDistribution(list(taus), seed=int(seed)),
    "isotonic": lambda taus, seed: IsotonicPitDistribution(list(taus)),
    "stack": lambda taus, seed: StackedDistribution(list(taus), seed=int(seed)),
    "qar": lambda taus, seed: _QarOneStepHead(taus),
    "regime": lambda taus, seed: RegimeDistribution(list(taus), seed=int(seed)),
    "fhs_skew": lambda taus, seed: FhsSkewDistribution(list(taus)),
    "lgbm_q2": lambda taus, seed: LGBMQ2Distribution(list(taus), seed=int(seed)),
    "conf_t": lambda taus, seed: ConformalTDistribution(list(taus)),
    "hstep_t": lambda taus, seed: _HStepOneStepHead(taus, "student_t"),
    "hstep_emp": lambda taus, seed: _HStepOneStepHead(taus, "empirical"),
}


def resolve_shard_generators(names: Iterable[str] | None = None) -> dict[str, ShardGenerator]:
    """Resolve shard names against ``SHARD_GENERATORS`` (default: all)."""
    if names is None:
        return dict(SHARD_GENERATORS)
    resolved: dict[str, ShardGenerator] = {}
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        if name not in SHARD_GENERATORS:
            raise ValueError(f"unknown fleet shard {name!r}")
        resolved[name] = SHARD_GENERATORS[name]
    if not resolved:
        raise ValueError("fleet requires at least one shard")
    return resolved


def _head_factory(name: str, taus: Sequence[float], seed: int) -> HeadFactory:
    def _factory() -> Any:
        return FLEET_HEAD_REGISTRY[name](taus, seed)

    return _factory


def fleet_head_factories(
    taus: Sequence[float],
    seed: int,
    names: Iterable[str] | None = None,
) -> dict[str, HeadFactory]:
    """Build ``{name: factory}`` for the fleet from ``FLEET_HEAD_REGISTRY``."""
    chosen = list(FLEET_HEAD_REGISTRY) if names is None else [str(s).strip() for s in names]
    chosen = [name for name in chosen if name]
    if not chosen:
        raise ValueError("fleet requires at least one head")
    unknown = sorted(set(chosen) - set(FLEET_HEAD_REGISTRY))
    if unknown:
        raise ValueError(f"unknown fleet head(s): {', '.join(unknown)}")
    return {name: _head_factory(name, taus, seed) for name in chosen}


def _pinball_key(tau: float) -> str:
    return f"pinball_{tau:g}"


def _coverage_key(level: float) -> str:
    return f"coverage_{int(round(level * 100))}"


def _central_interval_index(taus: Array, level: float) -> tuple[int, int] | None:
    """Indices of the central ``level`` interval on the tau grid, else None."""
    lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    i = np.flatnonzero(np.isclose(taus, lo, atol=1e-9))
    j = np.flatnonzero(np.isclose(taus, hi, atol=1e-9))
    if i.size == 0 or j.size == 0:
        return None
    return int(i[0]), int(j[0])


def _score_row(
    shard: SyntheticShard,
    shard_seed: int,
    name: str,
    factory: HeadFactory,
    n_train: int,
    n_eval: int,
    taus: Array,
    coverage_index: dict[float, tuple[int, int] | None],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "shard": shard.name,
        "model": name,
        "family": "distribution",
        "status": "ok",
        "error": None,
        "n_train": n_train,
        "n_eval": n_eval,
        "seed": shard_seed,
        "crps": None,
        "pit_ks": None,
        "pit_ks_p": None,
        **{_coverage_key(level): None for level in COVERAGE_LEVELS},
        **{_pinball_key(float(t)): None for t in taus},
    }
    try:
        model = factory()
        model.fit(shard.x[:n_train], shard.y[:n_train])
        meta = model.metadata()
        row["family"] = str(meta.family)
        row["model_version"] = str(getattr(meta, "version", "unknown") or "unknown")
        row["model_head"] = str(getattr(meta, "name", "") or "")
        if getattr(model, "fleet_lagged_predict", False):
            # One-step series heads consume the observed lag-1 return as the
            # predict matrix: row i's features are y[n_train+i-1], already
            # observed at that forecast origin — no lookahead.
            lag_x = shard.y[n_train - 1 : n_train + n_eval - 1].reshape(-1, 1)
            q = np.asarray(model.predict(lag_x), dtype=float)
        else:
            q = np.asarray(model.predict(shard.x[n_train : n_train + n_eval]), dtype=float)
        if q.ndim != 2 or q.shape[0] != n_eval or q.shape[1] != taus.size:
            raise ValueError(f"predict returned shape {q.shape}; expected ({n_eval}, {taus.size})")
        if not np.isfinite(q).all():
            raise ValueError("predict returned non-finite quantiles")
        if np.any(np.diff(q, axis=1) < 0.0):
            raise ValueError("predict returned crossing quantiles")
        y_eval = shard.y[n_train : n_train + n_eval]
        row["crps"] = crps_from_quantiles(y_eval, q, taus)
        ks, ks_p = pit_ks(pit_values(y_eval, q, taus))
        row["pit_ks"] = ks
        # The usual KS p-value assumes independent PIT draws. Shards declare
        # serial dependence in their generator config; for those, suppress the
        # p-value (the KS statistic itself stays reported).
        row["pit_ks_p"] = None if shard.config.get("serial_dependence") else ks_p
        for level, pair in coverage_index.items():
            if pair is not None:
                row[_coverage_key(level)] = coverage(y_eval, q[:, pair[0]], q[:, pair[1]])
        for j, tau in enumerate(taus):
            row[_pinball_key(float(tau))] = mean_pinball(y_eval, q[:, j], float(tau))
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def run_distribution_fleet(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    n_train: int = 512,
    n_eval: int = 256,
    *,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Fit every head on each shard's leading slice; score the trailing slice.

    Scores are proper rules only: per-tau pinball, quantile CRPS, PIT
    Kolmogorov-Smirnov, and central 80/90% coverage. A head that fails to fit
    or predict is recorded as an ``error`` row (visible, never silent); the
    harness itself fails closed on degenerate arguments or degenerate shard
    output. Returns the results frame plus the unsealed receipt payload.
    """
    if not isinstance(factories, Mapping) or not factories:
        raise ValueError("fleet requires a nonempty mapping of head factories")
    if (
        isinstance(n_train, bool)
        or not isinstance(n_train, (int, np.integer))
        or n_train < 1
        or isinstance(n_eval, bool)
        or not isinstance(n_eval, (int, np.integer))
        or n_eval < 1
    ):
        raise ValueError("n_train and n_eval must be positive")
    n_train = int(n_train)
    n_eval = int(n_eval)
    tau_arr = np.asarray(list(taus), dtype=float)
    if (
        tau_arr.size == 0
        or not np.isfinite(tau_arr).all()
        or np.any((tau_arr <= 0.0) | (tau_arr >= 1.0))
        or np.any(np.diff(tau_arr) <= 0.0)
    ):
        raise ValueError("taus must be a nonempty strictly increasing grid inside (0, 1)")
    resolved: Mapping[str, ShardGenerator]
    if shards is None:
        resolved = SHARD_GENERATORS
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        resolved = resolve_shard_generators(shards)
    if not resolved:
        raise ValueError("fleet requires at least one shard")
    for name in resolved:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("shard names must be nonempty strings")

    coverage_index = {level: _central_interval_index(tau_arr, level) for level in COVERAGE_LEVELS}
    n_shard = n_train + n_eval
    rows: list[dict[str, Any]] = []
    shard_meta: dict[str, Any] = {}
    for shard_index, (shard_name, generator) in enumerate(resolved.items()):
        shard_seed = int(seed) + shard_index
        shard = generator(n_shard, shard_seed)
        if not isinstance(shard, SyntheticShard):
            raise ValueError(f"shard {shard_name!r} did not return a SyntheticShard")
        y = np.asarray(shard.y, dtype=float).reshape(-1)
        x = np.asarray(shard.x, dtype=float)
        if shard.name != shard_name or shard.config.get("data_label") != "SYNTHETIC":
            raise ValueError(f"shard {shard_name!r} must match its name and SYNTHETIC label")
        if y.size != n_shard or x.ndim != 2 or x.shape[0] != n_shard:
            raise ValueError(
                f"shard {shard_name!r} produced {y.size} rows; needs exactly {n_shard}"
            )
        if not np.isfinite(y).all() or not np.isfinite(x).all():
            raise ValueError(f"shard {shard_name!r} produced non-finite data")
        shard = SyntheticShard(shard.name, x, y, dict(shard.config))
        shard_meta[shard_name] = {
            "n": int(y.size),
            "seed": shard_seed,
            "x_sha256": hash_bytes(x.tobytes()),
            "y_sha256": hash_bytes(y.tobytes()),
            "config": shard.config,
        }
        for model_name, factory in factories.items():
            rows.append(
                _score_row(
                    shard, shard_seed, model_name, factory, n_train, n_eval, tau_arr, coverage_index
                )
            )

    model_versions: dict[str, dict[str, str]] = {}
    for row in rows:
        version = row.pop("model_version", None)
        head = row.pop("model_head", None)
        model_versions.setdefault(
            str(row["model"]),
            {
                "head": str(head) if head else str(row["model"]),
                "version": str(version) if version else "unknown",
            },
        )
    for model_name, blob in model_versions.items():
        if blob["version"] == "unknown":
            try:
                meta = factories[model_name]().metadata()
                head = getattr(meta, "name", None)
                version = getattr(meta, "version", None)
                if head:
                    blob["head"] = str(head)
                if version:
                    blob["version"] = str(version)
            except Exception:
                pass

    columns = [
        "shard",
        "model",
        "family",
        "status",
        "error",
        "n_train",
        "n_eval",
        "seed",
        "crps",
        "pit_ks",
        "pit_ks_p",
        *[_coverage_key(level) for level in COVERAGE_LEVELS],
        *[_pinball_key(float(t)) for t in tau_arr],
    ]
    schema = {
        **{key: pl.String for key in ("shard", "model", "family", "status", "error")},
        **{key: pl.Int64 for key in ("n_train", "n_eval", "seed")},
        **{
            key: pl.Float64
            for key in columns
            if key
            not in {"shard", "model", "family", "status", "error", "n_train", "n_eval", "seed"}
        },
    }
    frame = pl.DataFrame(rows, schema=schema, orient="row").select(columns)

    inputs_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "shards": {
                    name: {"x_sha256": meta["x_sha256"], "y_sha256": meta["y_sha256"]}
                    for name, meta in shard_meta.items()
                },
                "models": sorted(str(k) for k in factories),
                "model_versions": model_versions,
                "taus": [float(t) for t in tau_arr],
                "n_train": n_train,
                "n_eval": n_eval,
                "seed": int(seed),
            }
        )
    )
    receipt: dict[str, Any] = {
        "schema": FLEET_EVAL_SCHEMA,
        "kind": "distribution_fleet_eval",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "seed": int(seed),
        "n_train": n_train,
        "n_eval": n_eval,
        "taus": [float(t) for t in tau_arr],
        "models": sorted(str(k) for k in factories),
        "model_versions": model_versions,
        "shards": shard_meta,
        "inputs_sha256": inputs_sha256,
        "n_rows": len(rows),
        "n_error_rows": sum(1 for row in rows if row["status"] != "ok"),
        "results": rows,
    }
    return frame, receipt


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a complete immutable text artifact without replacing an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"receipt path is a symlink: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"receipt already exists with different content: {path}")
        return
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.is_symlink() or path.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"receipt already exists with different content: {path}"
                ) from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def fleet_v1_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Fail-closed contract for a ``fleet_eval.v1`` payload (writer + verifier)."""
    research_blob = {key: value for key, value in receipt.items() if key != "live_pnl_claim"}
    errors: list[str] = []
    if receipt.get("schema") != FLEET_EVAL_SCHEMA:
        errors.append("schema_not_fleet_eval_v1")
    if receipt.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if receipt.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not isinstance(receipt.get("results"), list) or not receipt["results"]:
        errors.append("results_missing_or_empty")
    if not family_blob_forbidden_metrics_absent(research_blob):
        errors.append("forbidden_metric_keys")
    return errors


def fleet_dataset_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The dataset identity bound by a v2 ``dataset_hash``: shard content hashes."""
    shards = receipt.get("shards")
    if not isinstance(shards, Mapping):
        raise ValueError("fleet receipt has no shards block")
    dataset: dict[str, Any] = {}
    for name, meta in shards.items():
        if not isinstance(meta, Mapping):
            raise ValueError(f"fleet shard {name!r} metadata is not an object")
        dataset[str(name)] = {
            "x_sha256": meta.get("x_sha256"),
            "y_sha256": meta.get("y_sha256"),
        }
    return dataset


def fleet_params(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The run parameters bound by a v2 ``params_hash``."""
    shards = receipt.get("shards")
    return {
        "models": receipt.get("models"),
        "taus": receipt.get("taus"),
        "n_train": receipt.get("n_train"),
        "n_eval": receipt.get("n_eval"),
        "seed": receipt.get("seed"),
        "shards": sorted(str(name) for name in shards) if isinstance(shards, Mapping) else None,
    }


def fleet_v1_verdict(receipt: Mapping[str, Any]) -> str:
    """pass iff every fleet row scored without error; a recorded error is a fail."""
    n_error_rows = receipt.get("n_error_rows")
    return "pass" if n_error_rows == 0 else "fail"


def fleet_receipt_v2(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a ``fleet_eval.v1`` payload in the unified ``receipt.v2`` envelope.

    The v1 payload is embedded verbatim under ``payload``; the envelope binds
    the shard data digests, run params, this module's source hash, and the
    loaded numeric stack. Validates the v1 contract first — a malformed v1
    receipt is never wrapped.
    """
    if fleet_v1_contract_errors(receipt):
        raise ValueError("fleet receipt violates its synthetic research contract")
    return build_receipt_v2(
        kind=str(receipt["kind"]),
        data_label=str(receipt["data_label"]),
        dataset=fleet_dataset_identity(receipt),
        params=fleet_params(receipt),
        code_files=(Path(__file__),),
        verdict=fleet_v1_verdict(receipt),
        payload=dict(receipt),
        generated_at=str(receipt["generated_at"]),
        revision=str(receipt["git_revision"]),
    )


def fleet_v2_consistency_errors(envelope: Mapping[str, Any]) -> list[str]:
    """Re-derive a fleet receipt.v2 envelope's bound digests from its payload."""
    errors: list[str] = []
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    contract_errors = fleet_v1_contract_errors(payload)
    errors.extend(f"payload_{name}" for name in contract_errors)
    if contract_errors:
        return errors
    try:
        dataset = fleet_dataset_identity(payload)
    except ValueError as exc:
        return [*errors, f"payload_{exc}"]
    if hash_bytes(canonical_json_bytes(dataset)) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if hash_bytes(canonical_json_bytes(fleet_params(payload))) != envelope.get("params_hash"):
        errors.append("params_hash_mismatch")
    if fleet_v1_verdict(payload) != envelope.get("verdict"):
        errors.append("verdict_mismatch")
    return errors


def write_fleet_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a fleet receipt and write ``receipts/fleet_eval_<hash>.json``.

    The filename hash is the sha256 of the canonical receipt payload; the same
    digest is embedded as ``receipt_sha256`` (mirroring the real_benchmark
    seal convention). The write is atomic. ``receipt_version=2`` wraps the v1
    payload in the unified ``receipt.v2`` envelope before sealing.
    """
    if receipt_version == 1:
        if fleet_v1_contract_errors(receipt):
            raise ValueError("fleet receipt violates its synthetic research contract")
        body: Mapping[str, Any] = receipt
    elif receipt_version == 2:
        body = fleet_receipt_v2(receipt)
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    payload = seal_receipt(body)
    path = Path(receipts_dir) / f"fleet_eval_{payload['receipt_sha256'][:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
