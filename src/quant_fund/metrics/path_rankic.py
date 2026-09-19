"""Kronos-paper path RankIC: per-sample Spearman of predicted vs realized OHLC.

This is not Dipcatcher date-level cross-sectional IC. A path RankIC win does
not size the book. Research-only; no Sharpe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.scoring import rank_ic

Array = NDArray[np.float64]
OHLC_NAMES = ("open", "high", "low", "close")


def channel_rankic(pred: Array, realized: Array) -> dict[str, float]:
    """Spearman RankIC per OHLC channel, then the paper's unweighted mean."""
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(realized, dtype=np.float64)
    if p.ndim != 2 or r.ndim != 2 or p.shape != r.shape or p.shape[1] < 4:
        raise ValueError("pred and realized must be matching (H, >=4) arrays")
    channels: dict[str, float] = {}
    values: list[float] = []
    for i, name in enumerate(OHLC_NAMES):
        value = float(rank_ic(p[:, i], r[:, i]))
        channels[name] = value
        if np.isfinite(value):
            values.append(value)
    channels["mean"] = float(np.mean(values)) if values else float("nan")
    return channels


def mean_path_rankic(rows: list[dict[str, float]]) -> dict[str, Any]:
    """Pool per-window channel RankICs. Empty → honest NaNs."""
    if not rows:
        return {
            "n": 0,
            "mean": float("nan"),
            **{name: float("nan") for name in OHLC_NAMES},
        }
    out: dict[str, Any] = {"n": int(len(rows))}
    for name in (*OHLC_NAMES, "mean"):
        arr = np.array([row[name] for row in rows], dtype=float)
        finite = arr[np.isfinite(arr)]
        out[name] = float(np.mean(finite)) if finite.size else float("nan")
        out[f"{name}_n"] = int(finite.size)
    return out
