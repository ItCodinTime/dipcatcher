"""Candle mean_imbalance_top ∈[-1,1] wire + shape_columns_ensured ⇒ shape rates."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    mean_imbalance_top_honesty_errors,
    northset_shape_columns_ensured_rates_honesty_errors,
)


def test_candle_mean_imbalance_top_unit() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert "mean_imbalance_top" in receipt
    assert mean_imbalance_top_honesty_errors(receipt) == []
    bad = dict(receipt)
    bad["mean_imbalance_top"] = 1.5
    assert "mean_imbalance_top_out_of_unit_interval" in mean_imbalance_top_honesty_errors(bad)


def test_shape_columns_ensured_requires_shape_rates() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert receipt.get("shape_columns_ensured") is True
    assert northset_shape_columns_ensured_rates_honesty_errors(receipt) == []

    missing = dict(receipt)
    del missing["depth_shape_finite_rate"]
    assert "depth_shape_finite_rate_missing_while_shape_columns_ensured" in (
        northset_shape_columns_ensured_rates_honesty_errors(missing)
    )

    # False ⇒ skip
    assert (
        northset_shape_columns_ensured_rates_honesty_errors({"shape_columns_ensured": False}) == []
    )


def test_verify_wires_candle_mit_and_shape_ensured() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert 'mean_imbalance_top_honesty_errors(families.get("candle_order_book"))' in src
    assert "northset_shape_columns_ensured_rates_honesty_errors" in src
