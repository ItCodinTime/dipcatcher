"""Candle_order_book stamped sizing soft-verify: min_names / n_bars / n_fused / n_scored."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure import bench_candle_order_book
from quant_fund.research.catalog import candle_order_book_sizing_honesty_errors


def test_synth_candle_sizing_honesty_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars(),
        book=None,
        depth=5,
        seed=7,
        label="SYNTHETIC",
        min_names=3,
    )
    assert int(receipt["min_names"]) >= 1
    assert int(receipt["n_scored"]) <= int(receipt["n_fused"]) <= int(receipt["n_bars"])
    assert candle_order_book_sizing_honesty_errors(receipt) == []


def test_min_names_lt_one_fail_closed() -> None:
    assert "candle_min_names_lt_one_or_not_int" in candle_order_book_sizing_honesty_errors(
        {"family": "candle_order_book", "min_names": 0}
    )


def test_n_scored_gt_n_fused_fail_closed() -> None:
    assert "candle_n_scored_gt_n_fused" in candle_order_book_sizing_honesty_errors(
        {"family": "candle_order_book", "n_fused": 10, "n_scored": 11}
    )


def test_n_fused_gt_n_bars_fail_closed() -> None:
    assert "candle_n_fused_gt_n_bars" in candle_order_book_sizing_honesty_errors(
        {"family": "candle_order_book", "n_bars": 10, "n_fused": 12}
    )


def test_northset_family_skipped() -> None:
    assert (
        candle_order_book_sizing_honesty_errors(
            {"family": "northset", "min_names": 0, "n_fused": 1, "n_scored": 99}
        )
        == []
    )


def test_depth_lt_one_fail_closed() -> None:
    assert "candle_depth_lt_one_or_not_int" in candle_order_book_sizing_honesty_errors(
        {"family": "candle_order_book", "depth": 0}
    )


def test_synth_candle_depth_stamped() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars(),
        depth=5,
        min_names=3,
    )
    assert int(receipt["depth"]) >= 1
    assert candle_order_book_sizing_honesty_errors(receipt) == []


def test_verify_wires_candle_sizing_honesty() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_order_book_sizing_honesty_errors" in src
    assert 'families.get("candle_order_book")' in src
