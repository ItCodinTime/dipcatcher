"""Causal DLinear path baseline (Zeng et al., 2023). Research-only.

A moving-average trend plus a linear residual, fit only on history strictly
at or before as-of. This is the Kronos paper's non-pretrained TSFM baseline
that Dipcatcher can run without Hub weights. PatchTST / iTransformer /
TimesFM / Chronos / Time-MoE remain skipped unless local checkpoints exist.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def dlinear_forecast(
    history: Array,
    pred_len: int,
    *,
    kernel: int = 7,
) -> Array:
    """Forecast ``pred_len`` steps from a causal ``(T, C)`` window.

    Each channel is decomposed as a trailing moving average (trend) plus
    remainder. Trend is extrapolated with a linear fit on the MA series;
    remainder is held at the last observed seasonal residual. No future bars
    enter the fit.
    """
    x = np.asarray(history, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("history must be (T, C) with T>=2")
    if pred_len < 1:
        raise ValueError("pred_len must be positive")
    t_len, n_channels = x.shape
    out = np.full((pred_len, n_channels), np.nan, dtype=np.float64)
    k = max(2, min(int(kernel), t_len))
    weights = np.ones(k, dtype=np.float64) / float(k)
    idx = np.arange(t_len, dtype=np.float64)
    for channel in range(n_channels):
        series = x[:, channel]
        finite = np.isfinite(series)
        if int(finite.sum()) < 3:
            last = series[finite][-1] if finite.any() else np.nan
            out[:, channel] = last
            continue
        filled = series.copy()
        # Causal fill: look-behind only.
        last_val = filled[finite][0]
        for i in range(t_len):
            if np.isfinite(filled[i]):
                last_val = filled[i]
            else:
                filled[i] = last_val
        trend = np.convolve(filled, weights, mode="valid")
        if trend.size < 2:
            out[:, channel] = filled[-1]
            continue
        trend_x = idx[-trend.size :]
        slope, intercept = np.polyfit(trend_x, trend, 1)
        remainder = filled[-1] - trend[-1]
        last_i = float(idx[-1])
        for step in range(pred_len):
            out[step, channel] = intercept + slope * (last_i + step + 1.0) + remainder
    return out
