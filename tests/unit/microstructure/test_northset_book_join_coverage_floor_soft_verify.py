"""book_join_coverage_floor ∈ [0,1] via floors soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_shape_and_session_l2_floors_honesty_errors


def test_synth_book_join_coverage_floor_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "book_join_coverage_floor" in receipt
    assert 0.0 <= float(receipt["book_join_coverage_floor"]) <= 1.0
    assert northset_shape_and_session_l2_floors_honesty_errors(receipt) == []


def test_book_join_coverage_floor_oob_fail_closed() -> None:
    assert "book_join_coverage_floor_out_of_unit_interval" in (
        northset_shape_and_session_l2_floors_honesty_errors({"book_join_coverage_floor": 1.5})
    )


def test_book_join_coverage_floor_negative_fail_closed() -> None:
    assert "book_join_coverage_floor_out_of_unit_interval" in (
        northset_shape_and_session_l2_floors_honesty_errors({"book_join_coverage_floor": -0.1})
    )


def test_book_join_coverage_floor_inf_fail_closed() -> None:
    assert "book_join_coverage_floor_out_of_unit_interval" in (
        northset_shape_and_session_l2_floors_honesty_errors(
            {"book_join_coverage_floor": float("inf")}
        )
    )
