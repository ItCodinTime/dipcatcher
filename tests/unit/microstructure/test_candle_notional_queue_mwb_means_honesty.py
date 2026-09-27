"""Candle means: notional_imbalance ∈[-1,1]; queue_priority ∈[0,1]; MWB ∈[0,1]."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import (
    mean_microprice_weight_balance_honesty_errors,
    mean_notional_imbalance_honesty_errors,
    mean_queue_priority_honesty_errors,
)


def test_fail_closed() -> None:
    assert "mean_notional_imbalance_out_of_unit_interval" in (
        mean_notional_imbalance_honesty_errors({"mean_notional_imbalance": 1.5})
    )
    assert "mean_queue_priority_proxy_out_of_unit_interval" in (
        mean_queue_priority_honesty_errors({"mean_queue_priority_proxy": 1.2})
    )
    assert "mean_microprice_weight_balance_out_of_unit_interval" in (
        mean_microprice_weight_balance_honesty_errors({"mean_microprice_weight_balance": -0.1})
    )


def test_synth_candle_means() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert mean_notional_imbalance_honesty_errors(receipt) == []
    assert mean_queue_priority_honesty_errors(receipt) == []
    assert mean_microprice_weight_balance_honesty_errors(receipt) == []


def test_verify_wires_candle_queue_and_notional_mwb() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "mean_notional_imbalance_honesty_errors" in src
    assert 'families.get("candle_order_book")' in src
    assert src.count("mean_queue_priority_honesty_errors") >= 1
    assert (
        'mean_queue_priority_honesty_errors(\n            families.get("candle_order_book")' in src
    )
    assert (
        'mean_microprice_weight_balance_honesty_errors(\n            families.get("candle_order_book")'
        in src
    )
