"""Candle_order_book family / book_source / label stamp soft-verify."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure import bench_candle_order_book
from quant_fund.research.catalog import candle_order_book_family_provenance_honesty_errors


def test_synth_candle_family_provenance_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars(),
        min_names=3,
        label="SYNTHETIC",
    )
    assert receipt.get("family") == "candle_order_book"
    assert receipt.get("book_source")
    assert receipt.get("label")
    assert candle_order_book_family_provenance_honesty_errors(receipt) == []


def test_family_invalid_fail_closed() -> None:
    # family wrong but candle markers → still need family key == candle when family set
    # helper skips non-candle family entirely
    assert (
        candle_order_book_family_provenance_honesty_errors(
            {"family": "northset", "book_source": "", "label": ""}
        )
        == []
    )


def test_book_source_empty_fail_closed() -> None:
    assert "candle_book_source_empty_or_not_str" in (
        candle_order_book_family_provenance_honesty_errors(
            {"family": "candle_order_book", "book_source": " "}
        )
    )


def test_label_empty_fail_closed() -> None:
    assert "candle_label_empty_or_not_str" in (
        candle_order_book_family_provenance_honesty_errors(
            {"family": "candle_order_book", "label": ""}
        )
    )


def test_verify_wires_candle_family_provenance() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_order_book_family_provenance_honesty_errors" in src
