"""Shared synthetic fixtures for explainability tests (all labeled SYNTHETIC)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from quant_fund.models.base import ModelMeta

FEATURES = ["mom_20", "reversal_1", "vol_20", "signal_core", "noise_a", "noise_b"]


class LinearHead:
    """Minimal ForecastModel-conformant head fit by deterministic least squares."""

    def __init__(self, features: list[str] | None = None) -> None:
        self.cols = features or list(FEATURES)
        self.coef: NDArray[np.float64] = np.zeros(len(self.cols), dtype=float)
        self.fitted = False

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> LinearHead:
        coef, *_ = np.linalg.lstsq(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), rcond=None
        )
        self.coef = np.asarray(coef, dtype=float)
        self.fitted = True
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(x, dtype=float) @ self.coef

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="test",
            name="linear_head",
            version="v0",
            features=list(self.cols),
        )


def synthetic_xy(
    seed: int,
    n_rows: int = 400,
    n_features: int = len(FEATURES),
    weights: NDArray[np.float64] | None = None,
    noise: float = 0.3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Planted-signal design: ``y = X @ weights + eps`` (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_rows, n_features))
    w = np.zeros(n_features) if weights is None else np.asarray(weights, dtype=float)
    y = x @ w + rng.normal(0.0, noise, size=n_rows)
    return x, y


@pytest.fixture()
def planted_model() -> tuple[LinearHead, NDArray[np.float64], NDArray[np.float64]]:
    """Fitted synthetic model: ``signal_core`` (index 3) dominates the label."""
    weights = np.asarray([0.4, 0.2, -0.1, 4.0, 0.0, 0.0])
    x, y = synthetic_xy(2026, n_rows=600, weights=weights)
    model = LinearHead().fit(x, y)
    return model, x, y
