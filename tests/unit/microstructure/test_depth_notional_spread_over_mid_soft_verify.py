"""Stamp + soft-verify depth / side-notional / spread_over_mid means."""

from __future__ import annotations

import inspect

from quant_fund.northset import benches
from quant_fund.research.catalog import (
    northset_depth_notional_spread_over_mid_honesty_errors,
)


def test_bench_stamps_depth_notional_spread_over_mid() -> None:
    src = inspect.getsource(benches.bench_northset)
    for key in (
        "mean_bid_depth",
        "mean_ask_depth",
        "mean_side_notional_proxy_bid",
        "mean_side_notional_proxy_ask",
        "mean_top_of_book_notional_proxy",
        "mean_spread_over_mid",
    ):
        assert f'"{key}"' in src


def test_depth_notional_honesty_ok() -> None:
    assert (
        northset_depth_notional_spread_over_mid_honesty_errors(
            {
                "mean_bid_depth": 100.0,
                "mean_ask_depth": 90.0,
                "mean_side_notional_proxy_bid": 1.0,
                "mean_side_notional_proxy_ask": 1.0,
                "mean_top_of_book_notional_proxy": 0.5,
                "mean_spread_over_mid": 0.0001,
            }
        )
        == []
    )


def test_depth_notional_honesty_negative() -> None:
    assert "mean_spread_over_mid_negative" in (
        northset_depth_notional_spread_over_mid_honesty_errors({"mean_spread_over_mid": -1e-9})
    )
