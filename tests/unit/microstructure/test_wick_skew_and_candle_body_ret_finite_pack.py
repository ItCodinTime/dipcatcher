"""wick_skew + candle_body_ret finite means (candle) and northset IC pack."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    candle_wick_skew_and_body_ret_means_honesty_errors,
    northset_candle_body_ret_ic_pack_honesty_errors,
    northset_wick_skew_ic_pack_honesty_errors,
)


def test_candle_means_finite_fail_closed() -> None:
    assert "mean_wick_skew_non_finite" in (
        candle_wick_skew_and_body_ret_means_honesty_errors(
            {"family": "candle_order_book", "mean_wick_skew": float("inf")}
        )
    )
    assert "mean_candle_body_ret_non_finite" in (
        candle_wick_skew_and_body_ret_means_honesty_errors(
            {"family": "candle_order_book", "mean_candle_body_ret": float("-inf")}
        )
    )


def test_synth_candle_and_northset_packs() -> None:
    candle = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_wick_skew_and_body_ret_means_honesty_errors(candle) == []
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    north = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_wick_skew_ic_pack_honesty_errors(north) == []
    assert northset_candle_body_ret_ic_pack_honesty_errors(north) == []
    assert northset_wick_skew_ic_pack_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS


def test_verify_wires_candle_wick_body() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_wick_skew_and_body_ret_means_honesty_errors" in src
