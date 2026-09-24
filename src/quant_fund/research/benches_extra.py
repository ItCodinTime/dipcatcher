"""Additional proper-score research benches (optional scorecard families).

These families report descriptive scientific diagnostics of the SYNTHETIC return
panel — never live-P&L or headline ratios:

- ``complexity``: information-theoretic irregularity of per-asset return paths
  (permutation entropy, sample entropy, spectral entropy, Lempel-Ziv).
- ``roughness``: self-similarity / roughness exponents (rescaled-range and DFA
  Hurst, Higuchi fractal dimension).
- ``serial_randomness``: random-walk / serial-independence tests (Lo-MacKinlay
  variance ratio, runs test, Ljung-Box).

Each bench returns a mapping of finite floats averaged across assets, keeping the
honesty guardrails intact (no sharpe/sortino/calmar/pnl/nav key tokens).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.entropy import (
    lempel_ziv_complexity,
    permutation_entropy,
    sample_entropy,
    spectral_entropy,
)
from quant_fund.metrics.fractal import dfa_hurst, higuchi_fd, hurst_rs
from quant_fund.metrics.serial import ljung_box, runs_test, variance_ratio_test

Array = NDArray[np.float64]


def _per_asset_returns(frame: pl.DataFrame, min_obs: int = 64) -> list[Array]:
    """Time-ordered ``ret_1`` arrays per security with at least ``min_obs`` points."""
    if "ret_1" not in frame.columns:
        return []
    if "security_id" in frame.columns:
        cols = ["security_id", "ret_1"]
        if "event_time" in frame.columns:
            cols.append("event_time")
        sub = frame.select(cols).drop_nulls()
        if "event_time" in sub.columns:
            sub = sub.sort(["security_id", "event_time"])
        series = []
        for _, part in sub.group_by("security_id", maintain_order=True):
            arr = part["ret_1"].to_numpy().astype(float)
            if arr.size >= min_obs and np.isfinite(arr).all():
                series.append(arr)
        return series
    arr = frame.select("ret_1").drop_nulls()["ret_1"].to_numpy().astype(float)
    return [arr] if arr.size >= min_obs and np.isfinite(arr).all() else []


def _avg(series: list[Array], fn: Callable[[Array], float]) -> float:
    vals = []
    for s in series:
        try:
            v = fn(s)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def bench_complexity(frame: pl.DataFrame) -> dict[str, float]:
    """Information-theoretic complexity of return paths (asset-averaged)."""
    series = _per_asset_returns(frame)
    if not series:
        return {}
    return {
        "permutation_entropy": _avg(
            series, lambda s: permutation_entropy(s, order=3)["normalized"]
        ),
        "sample_entropy": _avg(series, lambda s: sample_entropy(s, m=2)),
        "spectral_entropy": _avg(series, lambda s: spectral_entropy(s)),
        "lempel_ziv": _avg(series, lambda s: lempel_ziv_complexity(s)),
        "n_assets": float(len(series)),
    }


def bench_roughness(frame: pl.DataFrame) -> dict[str, float]:
    """Self-similarity / roughness exponents of return paths (asset-averaged)."""
    series = _per_asset_returns(frame, min_obs=128)
    if not series:
        return {}
    return {
        "hurst_rs": _avg(series, lambda s: hurst_rs(s)["hurst"]),
        "dfa_hurst": _avg(series, lambda s: dfa_hurst(s)["hurst"]),
        "higuchi_fd": _avg(series, lambda s: higuchi_fd(s)),
        "n_assets": float(len(series)),
    }


def bench_serial_randomness(frame: pl.DataFrame) -> dict[str, float]:
    """Random-walk / serial-independence diagnostics (asset-averaged)."""
    series = _per_asset_returns(frame)
    if not series:
        return {}
    return {
        "variance_ratio_q2": _avg(series, lambda s: variance_ratio_test(s, q=2)["vr"]),
        "variance_ratio_q2_pvalue": _avg(
            series, lambda s: variance_ratio_test(s, q=2)["pvalue_z2"]
        ),
        "runs_pvalue": _avg(series, lambda s: runs_test(s)["pvalue"]),
        "ljung_box_pvalue": _avg(series, lambda s: ljung_box(s, lag=10)["pvalue"]),
        "n_assets": float(len(series)),
    }
