"""shape_columns_ensured ↔ book_panel_path stamp consistency."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_shape_columns_ensured_book_panel_path_honesty_errors,
)


def test_synth_ensured_true_path_none_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert receipt.get("shape_columns_ensured") is True
    assert receipt.get("book_panel_path") in (None, "")
    assert northset_shape_columns_ensured_book_panel_path_honesty_errors(receipt) == []


def test_ensured_true_with_path_fail_closed() -> None:
    assert "book_panel_path_set_while_shape_columns_ensured" in (
        northset_shape_columns_ensured_book_panel_path_honesty_errors(
            {
                "family": "northset",
                "shape_columns_ensured": True,
                "book_panel_path": "/tmp/panel.parquet",
            }
        )
    )


def test_ensured_false_missing_path_fail_closed() -> None:
    assert "book_panel_path_missing_while_shape_columns_not_ensured" in (
        northset_shape_columns_ensured_book_panel_path_honesty_errors(
            {"family": "northset", "shape_columns_ensured": False}
        )
    )


def test_ensured_false_with_path_clean() -> None:
    assert (
        northset_shape_columns_ensured_book_panel_path_honesty_errors(
            {
                "family": "northset",
                "shape_columns_ensured": False,
                "book_panel_path": "/tmp/panel.parquet",
            }
        )
        == []
    )


def test_helper_registered() -> None:
    assert (
        northset_shape_columns_ensured_book_panel_path_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
