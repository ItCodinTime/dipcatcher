"""Candle mean_depth_imbalance∈[-1,1] wire + METRICS_REQUIRED keys finite when present."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import (
    mean_depth_imbalance_abs_honesty_errors,
    mean_depth_imbalance_honesty_errors,
    northset_metrics_required_keys_finite_when_present_honesty_errors,
)


def test_candle_mean_depth_imbalance_unit() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert "mean_depth_imbalance" in receipt
    assert mean_depth_imbalance_honesty_errors(receipt) == []
    assert mean_depth_imbalance_abs_honesty_errors(receipt) == []
    bad = dict(receipt)
    bad["mean_depth_imbalance"] = 1.5
    assert "mean_depth_imbalance_out_of_unit_interval" in mean_depth_imbalance_honesty_errors(bad)
    bad_abs = dict(receipt)
    bad_abs["mean_depth_imbalance_abs"] = -0.1
    assert "mean_depth_imbalance_abs_out_of_unit_interval" in (
        mean_depth_imbalance_abs_honesty_errors(bad_abs)
    )


def test_metrics_required_keys_finite_when_present() -> None:
    assert northset_metrics_required_keys_finite_when_present_honesty_errors({}) == []
    assert (
        northset_metrics_required_keys_finite_when_present_honesty_errors(
            {"mid": 100.0, "spread": 0.01}
        )
        == []
    )
    assert "spread_non_finite_metrics_required" in (
        northset_metrics_required_keys_finite_when_present_honesty_errors({"spread": float("nan")})
    )


def test_verify_wires_candle_depth_and_metrics_keys() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert 'mean_depth_imbalance_honesty_errors(families.get("candle_order_book"))' in src
    assert 'mean_depth_imbalance_abs_honesty_errors(families.get("candle_order_book"))' in src
    assert "northset_metrics_required_keys_finite_when_present_honesty_errors" in src
