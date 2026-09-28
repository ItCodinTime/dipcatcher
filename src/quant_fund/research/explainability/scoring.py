"""Proper-score adapters for explainability.

Honesty contract: attribution is always measured as degradation in a PROPER
score — pinball, CRPS, or Brier — never accuracy, Sharpe, Sortino, P&L, or
NAV. Every score here is a loss: lower is better, so a positive permutation
delta means the feature carries real predictive content.

``pinball(tau=0.5)`` is the default. A point forecast is treated as the
conditional median forecast, where pinball is strictly proper (and equals
half the mean absolute error — a defensible attribution target for the
ranking heads, whose ``predict`` returns a continuous score).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.probability import brier_score
from quant_fund.metrics.scoring import crps_from_quantiles, mean_pinball

PredictFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""Model prediction contract: ``predict(x) -> (n,) or (n, k) array``."""

LossFn = Callable[[NDArray[np.float64], NDArray[np.float64]], float]
"""Proper loss contract: ``loss(y_true, y_pred) -> float``, lower is better."""

DEFAULT_CRPS_TAUS: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass(frozen=True)
class ProperScoreSpec:
    """A named proper scoring rule.

    ``fn(y_true, y_pred)`` returns a scalar loss (lower is better).
    ``expects_matrix`` is True when ``predict`` must emit an ``(n, k)``
    matrix (e.g. a quantile grid for CRPS) instead of a point forecast.
    """

    name: str
    fn: LossFn
    expects_matrix: bool = False

    def __call__(self, y: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
        actual = np.asarray(y, dtype=float)
        forecast = np.asarray(pred, dtype=float)
        if actual.ndim != 1 or forecast.ndim not in (1, 2):
            raise ValueError(
                "score requires one-dimensional labels and vector or matrix predictions"
            )
        if forecast.shape[0] != actual.shape[0]:
            raise ValueError("prediction rows must match labels")
        if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
            raise ValueError("labels and predictions must be finite")
        if self.expects_matrix:
            if forecast.ndim != 2:
                raise ValueError("score requires matrix predictions")
        elif forecast.ndim == 2:
            if forecast.shape[1] != 1:
                raise ValueError("point score requires one prediction column")
            forecast = forecast[:, 0]
        if self.name == "brier":
            if not np.isin(actual, (0.0, 1.0)).all():
                raise ValueError("Brier labels must be binary")
            if np.any((forecast < 0.0) | (forecast > 1.0)):
                raise ValueError("Brier predictions must be probabilities in [0, 1]")
        value = float(self.fn(actual, forecast))
        if not math.isfinite(value):
            raise ValueError("proper score must be finite")
        return value


def pinball(tau: float = 0.5) -> ProperScoreSpec:
    """Pinball (quantile) loss at level ``tau``; a point forecast is the τ-quantile.

    At ``tau=0.5`` this is half the MAE — strictly proper for the conditional
    median, which is the natural functional for point-forecast heads.
    """
    tau = float(tau)
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    return ProperScoreSpec(
        name=f"pinball_tau{tau:g}",
        fn=lambda y, pred: mean_pinball(y, pred, tau),
    )


def crps_quantiles(taus: tuple[float, ...] = DEFAULT_CRPS_TAUS) -> ProperScoreSpec:
    """Riemann-sum CRPS (Gneiting–Raftery) for heads emitting an ``(n, k)`` quantile grid."""
    grid = np.asarray(taus, dtype=float).ravel()
    if grid.size < 2:
        raise ValueError("crps_quantiles requires at least two quantile levels")
    if not np.isfinite(grid).all() or np.any(grid <= 0.0) or np.any(grid >= 1.0):
        raise ValueError("quantile levels must be finite and strictly inside (0, 1)")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("quantile levels must be strictly increasing")
    return ProperScoreSpec(
        name="crps_quantiles:" + ",".join(f"{t:g}" for t in grid.tolist()),
        fn=lambda y, pred: crps_from_quantiles(y, pred, grid),
        expects_matrix=True,
    )


def brier() -> ProperScoreSpec:
    """Brier score for probability heads; ``predict`` must emit P(y=1) in [0, 1]."""
    return ProperScoreSpec(
        name="brier",
        fn=lambda y, pred: brier_score(pred, y),
    )


def resolve_score(spec: ProperScoreSpec | str | LossFn | None) -> ProperScoreSpec:
    """Normalize a score specification.

    Accepts a :class:`ProperScoreSpec`, a string
    (``"pinball"``/``"pinball:0.25"``, ``"crps"``/``"crps:0.1,0.5,0.9"``,
    ``"brier"``), a callable ``(y, pred) -> float``, or ``None`` for the
    default ``pinball(0.5)``. Custom callables are the caller's honesty
    responsibility — they must be lower-is-better proper losses.
    """
    if spec is None:
        return pinball()
    if isinstance(spec, ProperScoreSpec):
        return spec
    if isinstance(spec, str):
        name, _, arg = spec.partition(":")
        key = name.strip().lower()
        if key == "pinball":
            return pinball(float(arg) if arg.strip() else 0.5)
        if key == "crps":
            levels = (
                tuple(float(tok) for tok in arg.split(",") if tok.strip())
                if arg.strip()
                else DEFAULT_CRPS_TAUS
            )
            return crps_quantiles(levels)
        if key == "brier":
            return brier()
        raise ValueError(f"unknown proper score {spec!r}")
    if callable(spec):
        return ProperScoreSpec(
            name=str(getattr(spec, "__name__", "custom")),
            fn=spec,
        )
    raise TypeError(f"unsupported score spec type: {type(spec).__name__}")


def predict_matrix(predict: PredictFn, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate ``predict`` and normalize the result to a 2-D float array."""
    out = np.asarray(predict(np.asarray(x, dtype=float)), dtype=float)
    if out.ndim == 1:
        return out.reshape(-1, 1)
    if out.ndim == 2:
        return out
    raise ValueError(f"predict returned unsupported shape {out.shape}")


def model_predict_fn(model: Any) -> PredictFn:
    """Adapt a ForecastModel-conformant head (``fit``/``predict``/``metadata``) to PredictFn."""
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise TypeError("model does not expose a callable predict(x)")
    return lambda x: np.asarray(predict(np.asarray(x, dtype=float)), dtype=float)
