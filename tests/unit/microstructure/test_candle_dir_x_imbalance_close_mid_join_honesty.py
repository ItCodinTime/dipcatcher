"""mean_candle_dir_x_imbalance ∈[-1,1]; close_mid_abs_rel ≥0; join_coverage on candle."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import (
    candle_join_coverage_and_chain_honesty_errors,
    mean_candle_dir_x_imbalance_honesty_errors,
    mean_close_mid_abs_rel_honesty_errors,
)


def test_fail_closed() -> None:
    assert "mean_candle_dir_x_imbalance_out_of_signed_unit" in (
        mean_candle_dir_x_imbalance_honesty_errors({"mean_candle_dir_x_imbalance": 1.5})
    )
    assert "mean_close_mid_abs_rel_negative" in (
        mean_close_mid_abs_rel_honesty_errors({"mean_close_mid_abs_rel": -1e-9})
    )


def test_synth_candle_pack() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert mean_candle_dir_x_imbalance_honesty_errors(receipt) == []
    assert mean_close_mid_abs_rel_honesty_errors(receipt) == []
    assert candle_join_coverage_and_chain_honesty_errors(receipt) == []


def test_verify_wires() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "mean_candle_dir_x_imbalance_honesty_errors" in src
    assert (
        'mean_close_mid_abs_rel_honesty_errors(\n            families.get("candle_order_book")'
        in src
    )
