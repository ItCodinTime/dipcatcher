"""Candle fuse: mean_ofi finite; mean_queue_imbalance ∈[-1,1] when stamped."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import candle_ofi_and_queue_imbalance_means_honesty_errors


def test_ofi_queue_fail_closed() -> None:
    base = {"family": "candle_order_book"}
    assert "mean_ofi_non_finite" in candle_ofi_and_queue_imbalance_means_honesty_errors(
        {**base, "mean_ofi": float("inf")}
    )
    assert "mean_queue_imbalance_out_of_signed_unit" in (
        candle_ofi_and_queue_imbalance_means_honesty_errors({**base, "mean_queue_imbalance": 1.5})
    )


def test_synth_candle_ofi_queue_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert "mean_ofi" in receipt and "mean_queue_imbalance" in receipt
    assert candle_ofi_and_queue_imbalance_means_honesty_errors(receipt) == []


def test_verify_wires_candle_ofi_queue_means() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_ofi_and_queue_imbalance_means_honesty_errors" in src
