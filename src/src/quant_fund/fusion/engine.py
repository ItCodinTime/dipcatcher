"""Transparent fusion and opt-in cross-fitted research stacking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.config.models import FusionConfig


@dataclass(frozen=True)
class CrossFittedStackingResult:
    """Research-only result from leakage-safe out-of-fold linear stacking."""

    weights: NDArray[np.float64]
    intercept: float
    oof_predictions: NDArray[np.float64]
    fold_ids: NDArray[np.int64]
    alpha: float


def _ridge_fit(
    x: NDArray[np.float64], y: NDArray[np.float64], alpha: float
) -> tuple[NDArray[np.float64], float]:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("stacker design and target shapes are incompatible")
    if x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("stacker requires at least two rows and one base prediction")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("stacker inputs must be finite")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("stacker alpha must be finite and positive")
    mean_x = x.mean(axis=0)
    mean_y = float(y.mean())
    centered = x - mean_x
    gram = centered.T @ centered
    weights = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), centered.T @ (y - mean_y))
    intercept = mean_y - float(mean_x @ weights)
    if not np.isfinite(weights).all() or not np.isfinite(intercept):
        raise ValueError("stacker fit produced non-finite parameters")
    return weights.astype(float), intercept


def cross_fitted_ridge_stack(
    base_predictions: NDArray[np.float64],
    target: NDArray[np.float64],
    folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    *,
    alpha: float = 1.0,
) -> CrossFittedStackingResult:
    """Fit an OOF ridge stacker without using a row's target in its prediction.

    ``folds`` must be chronological train/test index pairs. Warm-up observations
    may remain unscored; every test observation must be covered exactly once.
    Train/test overlap and duplicate test indices fail closed. This is a
    research diagnostic and does not authorize promotion.
    """
    x = np.asarray(base_predictions, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or not folds:
        raise ValueError("stacker requires a 2-D prediction matrix, target, and folds")
    n = x.shape[0]
    oof = np.full(n, np.nan, dtype=float)
    fold_ids = np.full(n, -1, dtype=np.int64)
    for fold_id, (train_raw, test_raw) in enumerate(folds):
        train = np.asarray(train_raw, dtype=np.int64).reshape(-1)
        test = np.asarray(test_raw, dtype=np.int64).reshape(-1)
        if train.size < 2 or test.size == 0:
            raise ValueError("each stacker fold needs training and test observations")
        if np.any(train < 0) or np.any(test < 0) or np.any(train >= n) or np.any(test >= n):
            raise ValueError("stacker fold index is out of bounds")
        if np.intersect1d(train, test).size:
            raise ValueError("stacker train/test fold overlap would leak targets")
        if np.unique(test).size != test.size or np.any(fold_ids[test] != -1):
            raise ValueError("stacker test observations must be covered exactly once")
        w, b = _ridge_fit(x[train], y[train], alpha)
        oof[test] = x[test] @ w + b
        fold_ids[test] = fold_id
    eligible = fold_ids >= 0
    if np.count_nonzero(eligible) < 2:
        raise ValueError("stacker folds must provide at least two scored observations")
    weights, intercept = _ridge_fit(x[eligible], y[eligible], alpha)
    return CrossFittedStackingResult(weights, intercept, oof, fold_ids, float(alpha))


def fuse_signals(
    alpha: NDArray[np.float64],
    confidence: NDArray[np.float64],
    regime_compat: NDArray[np.float64],
    predicted_risk: NDArray[np.float64],
    tail_penalty: NDArray[np.float64],
    liq_penalty: NDArray[np.float64],
    config: FusionConfig,
) -> NDArray[np.float64]:
    components = [
        np.asarray(value, dtype=float)
        for value in (alpha, confidence, regime_compat, predicted_risk, tail_penalty, liq_penalty)
    ]
    if any(value.ndim != 1 for value in components):
        raise ValueError("fusion components must be one-dimensional")
    if any(value.shape != components[0].shape for value in components[1:]):
        raise ValueError("fusion components must have matching lengths")
    if any(not np.isfinite(value).all() for value in components):
        raise ValueError("fusion components must contain only finite values")
    if np.any(components[3] < 0.0):
        raise ValueError("predicted_risk must be non-negative")
    # Negative confidence / regime compatibility would flip the sign of the
    # fused alpha; both are non-negative weights by contract.
    if np.any(components[1] < 0.0):
        raise ValueError("confidence must be non-negative")
    if np.any(components[2] < 0.0):
        raise ValueError("regime_compat must be non-negative")
    alpha, confidence, regime_compat, predicted_risk, tail_penalty, liq_penalty = components
    risk = np.clip(predicted_risk, 1e-8, None)
    raw = (
        config.alpha_weight
        * np.asarray(alpha, dtype=float)
        * (config.confidence_weight * np.asarray(confidence, dtype=float))
        * (config.regime_weight * np.asarray(regime_compat, dtype=float))
        / (config.risk_weight * risk)
    )
    return (
        raw
        - config.tail_penalty * np.asarray(tail_penalty, dtype=float)
        - config.liquidity_penalty * np.asarray(liq_penalty, dtype=float)
    )


_ABLATION_NAMES = frozenset(
    {"without_regime", "without_tail", "without_liquidity", "without_confidence"}
)


def ablation_inputs(
    name: str, arrays: dict[str, NDArray[np.float64]]
) -> dict[str, NDArray[np.float64]]:
    """Zero out one component for ablation. Unknown names fail closed."""
    if name not in _ABLATION_NAMES:
        raise ValueError(f"unknown ablation {name!r}; expected one of {sorted(_ABLATION_NAMES)}")
    out = {k: v.copy() for k, v in arrays.items()}
    if name == "without_regime":
        out["regime_compat"] = np.ones_like(out["regime_compat"])
    elif name == "without_tail":
        out["tail_penalty"] = np.zeros_like(out["tail_penalty"])
    elif name == "without_liquidity":
        out["liq_penalty"] = np.zeros_like(out["liq_penalty"])
    elif name == "without_confidence":
        out["confidence"] = np.ones_like(out["confidence"])
    return out
