"""Candle_order_book dgp/book_dgp ↔ data_source soft-verify (≠ northset twin)."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure import bench_candle_order_book
from quant_fund.research.catalog import candle_order_book_dgp_data_source_honesty_errors


def test_synth_candle_dgp_data_source_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars(),
        min_names=3,
        label="SYNTHETIC",
    )
    assert receipt.get("data_source") == "SYNTHETIC"
    assert receipt.get("book_dgp") == "synthetic_lob"
    assert candle_order_book_dgp_data_source_honesty_errors(receipt) == []


def test_dgp_mismatch_fail_closed() -> None:
    assert "candle_dgp_book_dgp_mismatch" in candle_order_book_dgp_data_source_honesty_errors(
        {
            "family": "candle_order_book",
            "dgp": "synthetic_lob",
            "book_dgp": "external_panel",
            "data_source": "SYNTHETIC",
        }
    )


def test_synthetic_requires_synthetic_lob() -> None:
    assert "candle_SYNTHETIC_data_source_book_dgp_not_synthetic_lob" in (
        candle_order_book_dgp_data_source_honesty_errors(
            {
                "family": "candle_order_book",
                "data_source": "SYNTHETIC",
                "book_dgp": "external_panel",
                "dgp": "external_panel",
            }
        )
    )


def test_northset_family_skipped() -> None:
    assert (
        candle_order_book_dgp_data_source_honesty_errors(
            {
                "family": "northset",
                "data_source": "SYNTHETIC",
                "book_dgp": "external_panel",
            }
        )
        == []
    )


def test_verify_wires() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_order_book_dgp_data_source_honesty_errors" in src
