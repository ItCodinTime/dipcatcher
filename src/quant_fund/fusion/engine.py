"""Transparent forecast fusion. Weights come from config, not a hidden meta-model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.config.models import FusionConfig


def fuse_signals(
    alpha: NDArray[np.float64],
    confidence: NDArray[np.float64],
    regime_compat: NDArray[np.float64],
    predicted_risk: NDArray[np.float64],
    tail_penalty: NDArray[np.float64],
    liq_penalty: NDArray[np.float64],
    config: FusionConfig,
) -> NDArray[np.float64]:
    risk = np.clip(np.asarray(predicted_risk, dtype=float), 1e-8, None)
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


def ablation_inputs(
    name: str, arrays: dict[str, NDArray[np.float64]]
) -> dict[str, NDArray[np.float64]]:
    """Zero out one component for ablation."""
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
