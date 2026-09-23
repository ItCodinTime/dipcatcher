"""AFML meta-label gate. Reduce-only: multiplier is in [0, 1].

López de Prado (2018): P(side correct | primary signal). Expanding-window
logistic with a one-row outcome delay. Never increases risk past the primary
sleeve. Not a live order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MetaLabelResult:
    primary_side: NDArray[np.float64]
    meta_prob: NDArray[np.float64]
    multiplier: NDArray[np.float64]
    threshold: float


def _as_1d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    series = np.asarray(values, dtype=float).reshape(-1)
    return series


def _design(
    primary_signal: NDArray[np.float64],
    realized_returns: NDArray[np.float64],
    features: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    sig = _as_1d(primary_signal)
    rets = _as_1d(realized_returns)
    if len(sig) != len(rets) or not len(sig):
        raise ValueError("finite, nonempty aligned signal and return vectors required")
    if not np.isfinite(sig).all() or not np.isfinite(rets).all():
        raise ValueError("finite, nonempty aligned signal and return vectors required")
    if features is None:
        volatility = np.array(
            [float(np.std(rets[max(0, i - 20) : i])) if i else 0.0 for i in range(len(sig))]
        )
        design = np.column_stack([np.abs(sig), sig, volatility])
    else:
        design = np.asarray(features, dtype=float)
        if design.ndim == 1:
            design = design[:, None]
        if (
            design.ndim != 2
            or design.shape[0] != len(sig)
            or not design.shape[1]
            or not np.isfinite(design).all()
        ):
            raise ValueError("features must be finite and aligned to signals")
    return sig, rets, design


def _predict_meta(
    training: NDArray[np.float64],
    labels: NDArray[np.float64],
    queries: NDArray[np.float64],
) -> NDArray[np.float64]:
    if len(training) < 10:
        return np.full(len(queries), 0.5)
    if np.all(labels == labels[0]):
        return np.full(len(queries), float(labels[0]))
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = np.column_stack([(training - mean) / scale, np.ones(len(training))])
    query = np.column_stack([(queries - mean) / scale, np.ones(len(queries))])
    beta = np.zeros(x.shape[1])
    regularizer = np.eye(len(beta)) * 1e-2
    for _ in range(40):
        probability = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
        hessian = x.T @ (x * (probability * (1.0 - probability))[:, None]) + regularizer
        step = np.linalg.solve(hessian, x.T @ (labels - probability) - regularizer @ beta)
        beta += step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    return 1.0 / (1.0 + np.exp(-np.clip(query @ beta, -30, 30)))


def metalabel_multiplier(
    probability: NDArray[np.float64],
    threshold: float = 0.55,
) -> NDArray[np.float64]:
    """Reduce-only size: 0 below tau, else max(2P-1, 0), clipped to [0, 1]."""
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    p = np.asarray(probability, dtype=float)
    scale = np.where(p >= float(threshold), np.maximum(2.0 * p - 1.0, 0.0), 0.0)
    return np.clip(scale, 0.0, 1.0)


def meta_label_gate(
    primary_signal: NDArray[np.float64],
    realized_returns: NDArray[np.float64],
    features: NDArray[np.float64] | None = None,
    threshold: float = 0.55,
) -> MetaLabelResult:
    """Expanding-window meta-label predictions with a one-row outcome delay."""
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    sig, rets, design = _design(primary_signal, realized_returns, features)
    side = np.sign(sig)
    labels = (side * rets > 0).astype(float)
    probabilities = np.full(len(sig), 0.5)
    for t in range(10, len(sig)):
        probabilities[t] = float(_predict_meta(design[:t], labels[:t], design[t : t + 1])[0])
    multiplier = metalabel_multiplier(probabilities, threshold)
    return MetaLabelResult(side, probabilities, multiplier, float(threshold))
