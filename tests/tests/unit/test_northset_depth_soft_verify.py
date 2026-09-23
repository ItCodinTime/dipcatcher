"""Always-on northset depth stamp soft-verify (≥1 int)."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_depth_honesty_errors,
)


def test_synth_northset_depth_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert int(receipt["depth"]) >= 1
    assert northset_depth_honesty_errors(receipt) == []


def test_depth_lt_one_fail_closed() -> None:
    assert "northset_depth_lt_one_or_not_int" in northset_depth_honesty_errors(
        {"family": "northset", "depth": 0}
    )


def test_depth_non_int_fail_closed() -> None:
    assert "northset_depth_lt_one_or_not_int" in northset_depth_honesty_errors(
        {"family": "northset", "depth": 1.5}
    )


def test_depth_inf_fail_closed() -> None:
    assert "northset_depth_non_finite" in northset_depth_honesty_errors(
        {"family": "northset", "depth": float("inf")}
    )


def test_candle_family_skipped() -> None:
    assert northset_depth_honesty_errors({"family": "candle_order_book", "depth": 0}) == []


def test_helper_registered() -> None:
    assert northset_depth_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
