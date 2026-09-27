"""Always-on northset dgp/book_dgp ↔ data_source soft-verify (≠ nest)."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_receipt_dgp_data_source_honesty_errors,
)


def test_synth_northset_dgp_data_source_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=20, seed=7).get_bars(),
        cfg,
    )
    assert receipt.get("data_source") == "SYNTHETIC"
    assert receipt.get("book_dgp") == "synthetic_lob"
    assert receipt.get("dgp") == "synthetic_lob"
    assert northset_receipt_dgp_data_source_honesty_errors(receipt) == []


def test_dgp_book_dgp_mismatch_fail_closed() -> None:
    errs = northset_receipt_dgp_data_source_honesty_errors(
        {
            "family": "northset",
            "dgp": "synthetic_lob",
            "book_dgp": "vendor_panel:x",
            "data_source": "SYNTHETIC",
        }
    )
    assert "northset_dgp_book_dgp_mismatch" in errs


def test_synthetic_data_source_requires_synthetic_lob() -> None:
    errs = northset_receipt_dgp_data_source_honesty_errors(
        {
            "family": "northset",
            "data_source": "SYNTHETIC",
            "book_dgp": "vendor_panel:x",
            "dgp": "vendor_panel:x",
        }
    )
    assert "northset_SYNTHETIC_data_source_book_dgp_not_synthetic_lob" in errs


def test_synthetic_dgp_requires_SYNTHETIC_data_source() -> None:
    errs = northset_receipt_dgp_data_source_honesty_errors(
        {
            "family": "northset",
            "book_dgp": "synthetic_lob",
            "data_source": "MIXED_SYNTHETIC_DERIVED",
        }
    )
    assert "northset_synthetic_dgp_data_source_not_SYNTHETIC" in errs


def test_candle_family_skipped() -> None:
    assert (
        northset_receipt_dgp_data_source_honesty_errors(
            {
                "family": "candle_order_book",
                "data_source": "SYNTHETIC",
                "book_dgp": "vendor_panel:x",
            }
        )
        == []
    )


def test_helper_registered() -> None:
    assert northset_receipt_dgp_data_source_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
