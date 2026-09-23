"""Stamp + soft-verify price slope / tick / top size / n_levels means."""

from __future__ import annotations

import inspect

from quant_fund.northset import benches
from quant_fund.research.catalog import (
    northset_price_slope_tick_top_levels_honesty_errors,
)

KEYS = (
    "mean_bid_log_price_slope",
    "mean_ask_log_price_slope",
    "mean_bid_mean_log_tick_spacing",
    "mean_ask_mean_log_tick_spacing",
    "mean_top_bid_size",
    "mean_top_ask_size",
    "mean_n_bid_levels",
    "mean_n_ask_levels",
)


def test_bench_stamps_price_slope_tick_top_levels() -> None:
    src = inspect.getsource(benches.bench_northset)
    for key in KEYS:
        assert f'"{key}"' in src


def test_price_slope_tick_honesty_ok() -> None:
    assert northset_price_slope_tick_top_levels_honesty_errors({k: 0.1 for k in KEYS}) == []


def test_price_slope_tick_honesty_bounds() -> None:
    assert "mean_top_ask_size_negative" in (
        northset_price_slope_tick_top_levels_honesty_errors({"mean_top_ask_size": -1})
    )
    assert "mean_ask_log_price_slope_non_finite" in (
        northset_price_slope_tick_top_levels_honesty_errors(
            {"mean_ask_log_price_slope": float("inf")}
        )
    )
