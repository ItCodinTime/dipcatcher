"""Always-on family / book_source stamp soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_family_book_source_honesty_errors,
)


def test_synth_family_book_source_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert receipt.get("family") == "northset"
    assert isinstance(receipt.get("book_source"), str) and receipt["book_source"]
    assert northset_family_book_source_honesty_errors(receipt) == []


def test_family_invalid_fail_closed() -> None:
    assert "northset_family_invalid" in northset_family_book_source_honesty_errors(
        {"family": "not_northset", "book_source": "synthetic_lob"}
    )


def test_book_source_empty_fail_closed() -> None:
    assert "northset_book_source_empty_or_not_str" in (
        northset_family_book_source_honesty_errors({"family": "northset", "book_source": "  "})
    )


def test_candle_family_skipped() -> None:
    assert (
        northset_family_book_source_honesty_errors(
            {"family": "candle_order_book", "book_source": ""}
        )
        == []
    )


def test_absent_skipped() -> None:
    assert northset_family_book_source_honesty_errors({}) == []


def test_label_empty_fail_closed() -> None:
    assert "northset_label_empty_or_not_str" in northset_family_book_source_honesty_errors(
        {"family": "northset", "label": "  "}
    )


def test_synth_label_nonempty() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert isinstance(receipt.get("label"), str) and receipt["label"].strip()
    assert northset_family_book_source_honesty_errors(receipt) == []


def test_helper_registered() -> None:
    assert northset_family_book_source_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
