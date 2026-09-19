"""Soft-verify join_coverage ∈(0,1] or [floor,1] for northset + candle_order_book."""

from __future__ import annotations

import math

from quant_fund.research.catalog import join_coverage_honesty_errors


def test_join_coverage_open_unit_interval_without_floor() -> None:
    assert join_coverage_honesty_errors({"join_coverage": 1.0}) == []
    assert join_coverage_honesty_errors({"join_coverage": 0.5}) == []
    assert join_coverage_honesty_errors({"join_coverage": float("nan")}) == []
    assert join_coverage_honesty_errors({}) == []
    # open at 0: exact zero fails fail-closed
    assert join_coverage_honesty_errors({"join_coverage": 0.0}) == [
        "join_coverage_outside_open_unit_interval_fail_closed"
    ]
    assert join_coverage_honesty_errors({"join_coverage": 1.01}) == [
        "join_coverage_outside_open_unit_interval_fail_closed"
    ]
    assert join_coverage_honesty_errors({"join_coverage": -0.1}) == [
        "join_coverage_outside_open_unit_interval_fail_closed"
    ]
    assert join_coverage_honesty_errors({"join_coverage": float("inf")}) == [
        "join_coverage_non_finite_fail_closed"
    ]


def test_join_coverage_floor_closed_interval() -> None:
    # With floor: [min_join, 1] — 0.5 ok, 0.4 below floor fails
    assert (
        join_coverage_honesty_errors({"join_coverage": 0.5, "book_join_coverage_floor": 0.5}) == []
    )
    assert (
        join_coverage_honesty_errors({"join_coverage": 1.0, "book_join_coverage_floor": 0.5}) == []
    )
    assert join_coverage_honesty_errors(
        {"join_coverage": 0.4, "book_join_coverage_floor": 0.5}
    ) == ["join_coverage_outside_floor_to_one_fail_closed"]
    assert join_coverage_honesty_errors(
        {"join_coverage": 1.01, "book_join_coverage_floor": 0.5}
    ) == ["join_coverage_outside_floor_to_one_fail_closed"]
    # min_join_coverage alias
    assert join_coverage_honesty_errors({"join_coverage": 0.75, "min_join_coverage": 0.5}) == []
    # Invalid floor falls back to (0, 1]
    assert join_coverage_honesty_errors(
        {"join_coverage": 0.0, "book_join_coverage_floor": float("nan")}
    ) == ["join_coverage_outside_open_unit_interval_fail_closed"]


def test_candle_and_northset_receipts_pass_join_coverage() -> None:
    from quant_fund.config.models import AppConfig
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.bench import bench_candle_order_book
    from quant_fund.northset.benches import bench_northset

    bars = SyntheticMarketProvider(n_assets=4, n_days=16, seed=42).get_bars()
    cob = bench_candle_order_book(bars, depth=5, seed=42, min_names=3)
    assert "join_coverage" in cob
    jc = float(cob["join_coverage"])
    assert math.isnan(jc) or 0.0 < jc <= 1.0
    assert join_coverage_honesty_errors(cob) == []

    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    ns = bench_northset(bars, cfg)
    assert "join_coverage" in ns
    assert "book_join_coverage_floor" in ns
    assert join_coverage_honesty_errors(ns) == []
