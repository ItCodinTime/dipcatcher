"""Fuse PIT book-age honesty on northset (mean/max ≥0; max ≥ mean)."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import book_age_seconds_honesty_errors


def test_synth_northset_book_age_fuse_honesty_clean() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(bars, cfg)
    assert "mean_book_age_seconds" in receipt
    assert "max_book_age_seconds" in receipt
    assert book_age_seconds_honesty_errors(receipt) == []


def test_book_age_max_lt_mean_fail_closed() -> None:
    errs = book_age_seconds_honesty_errors(
        {"mean_book_age_seconds": 10.0, "max_book_age_seconds": 1.0}
    )
    assert "max_book_age_seconds_lt_mean" in errs


def test_receipt_dispatcher_includes_book_age_helper() -> None:
    """Fuse residual stays covered via northset_receipt_honesty_errors fan-in."""
    from quant_fund.research.catalog import (
        NORTHSET_RECEIPT_HONESTY_HELPERS,
        book_age_seconds_honesty_errors,
    )

    assert book_age_seconds_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "northset_receipt_honesty_errors" in src
