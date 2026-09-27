"""Scenario generators the Monte Carlo engine can parallelize.

A generator is a pure function of ``(path_indices, shocks, seed)``. The engine
owns chunking, Philox or Sobol shocks, and the reduction. Bootstrap, copula,
and HMM engines plug in by implementing :class:`ScenarioGenerator`.

When ``accepts_external_shocks`` is true, ignore ``seed`` for randomness and
consume ``shocks``. When it is false, the engine only supports
``shock_mode='crude'`` and the generator must draw any randomness from
:func:`quant_fund.mc_engine.philox.philox_normals` (or uniforms) keyed by
``path_indices`` and ``seed``, with ``stream_id >= USER_STREAM_ID_MIN``.

``returns`` are simple portfolio returns with shape ``(n_paths, n_steps)``.
The numeraire starts at 1. A loss is ``1 - terminal level``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class ScenarioBatch:
    """One chunk of portfolio simple returns.

    ``control`` is a path-wise control variate with known mean ``control_mean``.
    ``log_importance_weight`` is added to any engine likelihood-ratio weight.
    """

    returns: FloatArray
    control: FloatArray | None = None
    control_mean: float | None = None
    log_importance_weight: FloatArray | None = None


class ScenarioGenerator(Protocol):
    """Plug-in point for GBM, bootstrap, copula, and HMM scenario engines.

    Attributes are read-only so frozen generators satisfy the protocol.
    """

    @property
    def name(self) -> str: ...

    @property
    def n_steps(self) -> int: ...

    @property
    def n_factors(self) -> int: ...

    @property
    def accepts_external_shocks(self) -> bool: ...

    @property
    def data_source(self) -> str: ...

    def spec_dict(self) -> dict[str, object]:
        """JSON-stable specification. Part of the checkpoint fingerprint."""
        ...

    def generate(
        self,
        path_indices: IntArray,
        *,
        shocks: FloatArray | None,
        seed: int,
    ) -> ScenarioBatch:
        """Return simple returns for these path indices, in the same order."""
        ...


def _as_1d(values: FloatArray, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < 1 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return arr


def _float_list(values: FloatArray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).ravel().tolist()]


def _cholesky_cov(covariance: FloatArray) -> tuple[FloatArray, FloatArray]:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.isfinite(cov).all():
        raise ValueError("covariance must be finite")
    # Symmetrize before the factor so an upper-triangle typo still has a defined matrix.
    cov = 0.5 * (cov + cov.T)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be symmetric positive definite") from exc
    variance = np.sum(chol * chol, axis=1)
    return np.asarray(chol, dtype=np.float64), np.asarray(variance, dtype=np.float64)


def correlate_shocks(shocks: FloatArray, chol: FloatArray, scale: float) -> FloatArray:
    """Lower-triangular mix. Elementwise, asset-major accumulation order.

    ``shocks`` is ``(n, t, k)`` independent normals. ``chol`` is lower
    triangular with ``chol @ chol.T = covariance``. The result is
    ``scale * shocks @ chol.T`` computed without a BLAS gemm so the sum order
    does not depend on the thread count.
    """
    z = np.asarray(shocks, dtype=np.float64)
    lower = np.asarray(chol, dtype=np.float64)
    if z.ndim != 3 or lower.ndim != 2 or z.shape[2] != lower.shape[0]:
        raise ValueError("shocks must have shape (n, t, k) matching chol")
    n_paths, n_steps, n_factors = z.shape
    out = np.zeros((n_paths, n_steps, n_factors), dtype=np.float64)
    factor = float(scale)
    for asset in range(n_factors):
        acc = np.zeros((n_paths, n_steps), dtype=np.float64)
        for source in range(asset + 1):
            acc = acc + z[:, :, source] * float(lower[asset, source])
        out[:, :, asset] = acc * factor
    return out


@dataclass(frozen=True)
class GbmPortfolioGenerator:
    """Static-weight rebalanced portfolio of correlated GBMs.

    Weights are long-only and sum to 1, so one-step portfolio simple returns
    stay strictly above -1 when each asset does. ``mu`` and ``covariance`` are
    annualized. ``dt`` is the step in years (default one trading day).
    """

    mu: FloatArray
    covariance: FloatArray
    weights: FloatArray
    n_steps: int
    dt: float = 1.0 / 252.0
    name: str = "gbm_rebalanced_portfolio"
    data_source: str = "SYNTHETIC"
    accepts_external_shocks: bool = True
    _chol: FloatArray = field(init=False, repr=False, compare=False)
    _variance: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mu = _as_1d(self.mu, "mu")
        weights = _as_1d(self.weights, "weights")
        if mu.shape != weights.shape:
            raise ValueError("mu and weights must have the same length")
        if np.any(weights < -1e-12):
            raise ValueError("GbmPortfolioGenerator weights must be long-only")
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if abs(total - 1.0) > 1e-8:
            raise ValueError("GbmPortfolioGenerator weights must sum to 1")
        weights = weights / total
        chol, variance = _cholesky_cov(self.covariance)
        if chol.shape[0] != mu.size:
            raise ValueError("covariance dimension must match mu")
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise ValueError("n_steps must be a positive int")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be positive")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "covariance",
            0.5
            * (
                np.asarray(self.covariance, dtype=np.float64)
                + np.asarray(self.covariance, dtype=np.float64).T
            ),
        )
        object.__setattr__(self, "_chol", chol)
        object.__setattr__(self, "_variance", variance)

    @property
    def n_factors(self) -> int:
        return int(self.mu.shape[0])

    def spec_dict(self) -> dict[str, object]:
        return {
            "type": "gbm_rebalanced_portfolio",
            "name": self.name,
            "data_source": self.data_source,
            "mu": _float_list(self.mu),
            "covariance": [
                _float_list(row) for row in np.asarray(self.covariance, dtype=np.float64)
            ],
            "weights": _float_list(self.weights),
            "n_steps": self.n_steps,
            "dt": float(self.dt),
        }

    def generate(
        self,
        path_indices: IntArray,
        *,
        shocks: FloatArray | None,
        seed: int,
    ) -> ScenarioBatch:
        del path_indices, seed
        if shocks is None:
            raise ValueError("GbmPortfolioGenerator requires external shocks")
        z = np.asarray(shocks, dtype=np.float64)
        if z.ndim != 3 or z.shape[1] != self.n_steps or z.shape[2] != self.n_factors:
            raise ValueError(f"shocks must have shape (n, {self.n_steps}, {self.n_factors})")
        if not np.isfinite(z).all():
            raise ValueError("shocks must be finite")
        diffusion = correlate_shocks(z, self._chol, math_sqrt(self.dt))
        drift = (self.mu - 0.5 * self._variance) * float(self.dt)
        log_ret = diffusion + drift.reshape(1, 1, -1)
        simple = np.exp(log_ret) - 1.0
        portfolio = np.zeros((z.shape[0], self.n_steps), dtype=np.float64)
        for asset in range(self.n_factors):
            portfolio = portfolio + simple[:, :, asset] * float(self.weights[asset])
        control = np.asarray(z[:, :, 0].sum(axis=1), dtype=np.float64)
        return ScenarioBatch(returns=portfolio, control=control, control_mean=0.0)


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


@dataclass(frozen=True)
class VolTargetStrategyGenerator:
    """Single-asset GBM with a causal trailing-volatility position.

    The position held over step ``t`` uses only asset simple returns before
    ``t``. The first ``lookback`` steps are flat (numeraire earns zero).
    Position is ``clip(target_vol / realized_vol, 0, max_gross)``.
    """

    mu: float
    sigma: float
    n_steps: int
    target_vol: float = 0.10
    lookback: int = 21
    max_gross: float = 1.0
    dt: float = 1.0 / 252.0
    min_vol: float = 1e-8
    name: str = "vol_target_strategy"
    data_source: str = "SYNTHETIC"
    accepts_external_shocks: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, int) or self.n_steps < 2:
            raise ValueError("n_steps must be an int >= 2")
        if (
            isinstance(self.lookback, bool)
            or not isinstance(self.lookback, int)
            or self.lookback < 2
        ):
            raise ValueError("lookback must be an int >= 2")
        if self.lookback >= self.n_steps:
            raise ValueError("lookback must be smaller than n_steps")
        for name, value in (
            ("mu", self.mu),
            ("sigma", self.sigma),
            ("target_vol", self.target_vol),
            ("dt", self.dt),
            ("min_vol", self.min_vol),
            ("max_gross", self.max_gross),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.sigma <= 0.0 or self.target_vol <= 0.0 or self.dt <= 0.0 or self.min_vol <= 0.0:
            raise ValueError("sigma, target_vol, dt, and min_vol must be positive")
        if self.max_gross < 0.0:
            raise ValueError("max_gross must be non-negative")

    @property
    def n_factors(self) -> int:
        return 1

    def spec_dict(self) -> dict[str, object]:
        return {
            "type": "vol_target_strategy",
            "name": self.name,
            "data_source": self.data_source,
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "n_steps": self.n_steps,
            "target_vol": float(self.target_vol),
            "lookback": self.lookback,
            "max_gross": float(self.max_gross),
            "dt": float(self.dt),
            "min_vol": float(self.min_vol),
        }

    def generate(
        self,
        path_indices: IntArray,
        *,
        shocks: FloatArray | None,
        seed: int,
    ) -> ScenarioBatch:
        del path_indices, seed
        if shocks is None:
            raise ValueError("VolTargetStrategyGenerator requires external shocks")
        z = np.asarray(shocks, dtype=np.float64)
        if z.ndim != 3 or z.shape[1] != self.n_steps or z.shape[2] != 1:
            raise ValueError(f"shocks must have shape (n, {self.n_steps}, 1)")
        if not np.isfinite(z).all():
            raise ValueError("shocks must be finite")
        shock = z[:, :, 0]
        drift_step = (float(self.mu) - 0.5 * float(self.sigma) ** 2) * float(self.dt)
        log_ret = drift_step + float(self.sigma) * math_sqrt(self.dt) * shock
        asset = np.exp(log_ret) - 1.0
        position = np.zeros_like(asset)
        annualizer = math_sqrt(1.0 / float(self.dt))
        for t in range(self.lookback, self.n_steps):
            window = asset[:, t - self.lookback : t]
            realized = np.std(window, axis=1, ddof=1) * annualizer
            realized = np.maximum(realized, float(self.min_vol))
            raw = float(self.target_vol) / realized
            position[:, t] = np.clip(raw, 0.0, float(self.max_gross))
        strategy = position * asset
        control = np.asarray(shock.sum(axis=1), dtype=np.float64)
        return ScenarioBatch(
            returns=np.asarray(strategy, dtype=np.float64),
            control=control,
            control_mean=0.0,
        )


@dataclass(frozen=True)
class IdentityShockGenerator:
    """Portfolio simple return equals factor-0 shock at each step.

    Reference generator for reproducibility and variance-reduction tests.
    Not a market model. With one step the loss is ``1 - (1 + shock)``. That
    map is odd in exact arithmetic. In float64 the pair mean is a residual,
    so the variance-reduction factor is large and is not forced to infinity.
    """

    n_steps: int
    n_factors: int = 1
    name: str = "identity_shock"
    data_source: str = "SYNTHETIC"
    accepts_external_shocks: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise ValueError("n_steps must be a positive int")
        if (
            isinstance(self.n_factors, bool)
            or not isinstance(self.n_factors, int)
            or self.n_factors < 1
        ):
            raise ValueError("n_factors must be a positive int")

    def spec_dict(self) -> dict[str, object]:
        return {
            "type": "identity_shock",
            "name": self.name,
            "data_source": self.data_source,
            "n_steps": self.n_steps,
            "n_factors": self.n_factors,
        }

    def generate(
        self,
        path_indices: IntArray,
        *,
        shocks: FloatArray | None,
        seed: int,
    ) -> ScenarioBatch:
        del path_indices, seed
        if shocks is None:
            raise ValueError("IdentityShockGenerator requires external shocks")
        z = np.asarray(shocks, dtype=np.float64)
        if z.ndim != 3 or z.shape[1] != self.n_steps or z.shape[2] != self.n_factors:
            raise ValueError(f"shocks must have shape (n, {self.n_steps}, {self.n_factors})")
        returns = np.asarray(z[:, :, 0], dtype=np.float64)
        control = np.asarray(z[:, :, 0].sum(axis=1), dtype=np.float64)
        return ScenarioBatch(returns=returns, control=control, control_mean=0.0)


def _spec_int(spec: dict[str, object], key: str) -> int:
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an int")
    return value


def _spec_float(spec: dict[str, object], key: str) -> float:
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a float")
    return float(value)


def generator_from_spec(spec: dict[str, object]) -> ScenarioGenerator:
    """Rebuild a built-in generator from its ``spec_dict``.

    Custom generators are not reconstructed here. Resume those through
    :func:`quant_fund.mc_engine.engine.resume_simulation` and pass the object.
    """
    kind = spec.get("type")
    if kind == "gbm_rebalanced_portfolio":
        mu = np.asarray(spec["mu"], dtype=np.float64)
        covariance = np.asarray(spec["covariance"], dtype=np.float64)
        weights = np.asarray(spec["weights"], dtype=np.float64)
        return GbmPortfolioGenerator(
            mu=mu,
            covariance=covariance,
            weights=weights,
            n_steps=_spec_int(spec, "n_steps"),
            dt=_spec_float(spec, "dt"),
            name=str(spec.get("name", "gbm_rebalanced_portfolio")),
        )
    if kind == "vol_target_strategy":
        return VolTargetStrategyGenerator(
            mu=_spec_float(spec, "mu"),
            sigma=_spec_float(spec, "sigma"),
            n_steps=_spec_int(spec, "n_steps"),
            target_vol=_spec_float(spec, "target_vol"),
            lookback=_spec_int(spec, "lookback"),
            max_gross=_spec_float(spec, "max_gross"),
            dt=_spec_float(spec, "dt"),
            min_vol=_spec_float(spec, "min_vol"),
            name=str(spec.get("name", "vol_target_strategy")),
        )
    if kind == "identity_shock":
        return IdentityShockGenerator(
            n_steps=_spec_int(spec, "n_steps"),
            n_factors=_spec_int(spec, "n_factors"),
            name=str(spec.get("name", "identity_shock")),
        )
    raise ValueError(
        f"cannot rebuild generator type {kind!r}; pass the generator to resume_simulation"
    )
